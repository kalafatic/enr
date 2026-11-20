package ai.enr.manager.download;

import ai.enr.core.json.JsonArray;
import ai.enr.core.json.JsonObject;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/**
 * HTTP download manager: background jobs with progress tracking, cancellation
 * and optional SHA-256 verification. Completed jobs notify listeners exactly
 * once. The manager is the single owner of download state.
 */
public final class DownloadManager implements AutoCloseable {

    public enum Status {
        RUNNING, COMPLETED, FAILED, CANCELLED, VERIFYING
    }

    public record Job(String id, String name, String url, Status status, long totalBytes,
                      long completedBytes, String digest, String error, String targetFile,
                      long startedAt, long finishedAt) {
    }

    public interface Listener {
        void onChanged(Job job);

        void onCompleted(Job job);
    }

    private final Map<String, Job> jobs = new ConcurrentHashMap<>();
    private final Map<String, InputStream> activeBodies = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newCachedThreadPool(r -> {
        Thread thread = new Thread(r, "enr-download");
        thread.setDaemon(true);
        return thread;
    });
    private final HttpClient http;
    private final List<Listener> listeners = new ArrayList<>();
    private volatile boolean closed = false;

    public DownloadManager() {
        this(HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.ALWAYS)
                .connectTimeout(java.time.Duration.ofSeconds(30))
                .build());
    }

    public DownloadManager(HttpClient http) {
        this.http = http;
    }

    public synchronized void addListener(Listener listener) {
        listeners.add(listener);
    }

    public String start(String name, String url, Path targetFile, String expectedDigest,
                        Consumer<Job> onComplete) {
        String id = UUID.randomUUID().toString();
        Job job = new Job(id, name, url, Status.RUNNING, -1, 0, expectedDigest, "",
                targetFile.toString(), System.currentTimeMillis(), 0);
        jobs.put(id, job);
        notifyChanged(job);
        executor.submit(() -> runJob(job, targetFile, onComplete));
        return id;
    }

    private void runJob(Job job, Path targetFile, Consumer<Job> onComplete) {
        try {
            Files.createDirectories(targetFile.getParent());
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(job.url()))
                    .GET()
                    .header("User-Agent", "enr/2.6")
                    .build();
            Path partFile = Path.of(targetFile + ".part");
            HttpResponse<InputStream> response = http.send(request,
                    HttpResponse.BodyHandlers.ofInputStream());
            if (response.statusCode() != 200) {
                fail(job, "HTTP " + response.statusCode());
                return;
            }
            long total = response.headers().firstValueAsLong("Content-Length").orElse(-1);
            if (total >= 0) {
                update(job, total, 0, Status.RUNNING, "");
            }
            AtomicLong completed = new AtomicLong();
            long[] lastFlush = {System.currentTimeMillis()};
            try (InputStream in = response.body();
                 OutputStream out = Files.newOutputStream(partFile)) {
                activeBodies.put(job.id(), in);
                byte[] buffer = new byte[64 * 1024];
                int read;
                while ((read = in.read(buffer)) >= 0) {
                    if (isCancelled(job)) {
                        break;
                    }
                    out.write(buffer, 0, read);
                    completed.addAndGet(read);
                    long now = System.currentTimeMillis();
                    if (now - lastFlush[0] > 300) {
                        lastFlush[0] = now;
                        update(job, total, completed.get(), Status.RUNNING, "");
                    }
                }
            } finally {
                activeBodies.remove(job.id());
            }
            if (isCancelled(job)) {
                Files.deleteIfExists(partFile);
                update(job, total, completed.get(), Status.CANCELLED, "cancelled by user");
                return;
            }
            if (job.digest() != null && !job.digest().isBlank()) {
                update(job, total, completed.get(), Status.VERIFYING, "");
                String actual = ai.enr.core.io.Hashing.sha256Hex(Files.readAllBytes(partFile));
                if (!job.digest().equals(actual)) {
                    Files.deleteIfExists(partFile);
                    fail(job, "SHA-256 mismatch: expected " + job.digest() + ", got " + actual);
                    return;
                }
            }
            Files.move(partFile, targetFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
            Job done = update(job, Files.size(targetFile), Files.size(targetFile), Status.COMPLETED, "");
            notifyCompleted(done);
            if (onComplete != null) {
                onComplete.accept(done);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            update(job, job.totalBytes(), job.completedBytes(), Status.CANCELLED, "interrupted");
        } catch (IOException e) {
            if (isCancelled(job)) {
                update(job, job.totalBytes(), job.completedBytes(), Status.CANCELLED, "cancelled by user");
            } else {
                fail(job, e.getMessage());
            }
        }
    }

    private Job update(Job job, long total, long completed, Status status, String error) {
        Job next = new Job(job.id(), job.name(), job.url(), status, total, completed,
                job.digest(), error, job.targetFile(), job.startedAt(),
                status == Status.COMPLETED || status == Status.FAILED || status == Status.CANCELLED
                        ? System.currentTimeMillis() : 0);
        jobs.put(job.id(), next);
        notifyChanged(next);
        return next;
    }

    private void fail(Job job, String error) {
        update(job, job.totalBytes(), job.completedBytes(), Status.FAILED, error);
    }

    public boolean cancel(String id) {
        Job job = jobs.get(id);
        if (job == null || job.status() != Status.RUNNING) {
            return false;
        }
        InputStream body = activeBodies.get(id);
        if (body != null) {
            try {
                body.close();
            } catch (IOException ignored) {
                // closing the body aborts the read loop
            }
        }
        return true;
    }

    public Optional<Job> get(String id) {
        return Optional.ofNullable(jobs.get(id));
    }

    public List<Job> list() {
        return jobs.values().stream().sorted(Comparator.comparingLong(Job::startedAt)).toList();
    }

    public JsonObject toJson() {
        JsonObject root = new JsonObject();
        JsonArray arr = new JsonArray();
        for (Job job : list()) {
            JsonObject obj = new JsonObject();
            obj.put("id", job.id());
            obj.put("name", job.name());
            obj.put("url", job.url());
            obj.put("status", job.status().name().toLowerCase());
            obj.put("total_bytes", job.totalBytes());
            obj.put("completed_bytes", job.completedBytes());
            obj.put("digest", job.digest() == null ? "" : job.digest());
            obj.put("error", job.error() == null ? "" : job.error());
            obj.put("target_file", job.targetFile());
            obj.put("started_at", job.startedAt());
            obj.put("finished_at", job.finishedAt());
            arr.add(obj);
        }
        root.put("downloads", arr);
        return root;
    }

    private boolean isCancelled(Job job) {
        Job current = jobs.get(job.id());
        return current == null || current.status() == Status.CANCELLED
                || current.status() == Status.FAILED;
    }

    private void notifyChanged(Job job) {
        for (Listener listener : List.copyOf(listeners)) {
            try {
                listener.onChanged(job);
            } catch (RuntimeException ignored) {
                // listeners must never break downloads
            }
        }
    }

    private void notifyCompleted(Job job) {
        for (Listener listener : List.copyOf(listeners)) {
            try {
                listener.onCompleted(job);
            } catch (RuntimeException ignored) {
                // listeners must never break downloads
            }
        }
    }

    @Override
    public void close() {
        closed = true;
        executor.shutdownNow();
    }

    public boolean isClosed() {
        return closed;
    }
}
