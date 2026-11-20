package ai.enr.manager.router;

import ai.enr.manager.logs.LogBuffer;
import ai.enr.manager.provider.Provider;
import ai.enr.manager.provider.ProviderRegistry;
import ai.enr.manager.store.ModelStore;

import java.util.ArrayList;
import java.util.List;

/**
 * Model router: resolves a logical model name to exactly one target — a
 * locally installed model or a model on an explicitly configured provider.
 * Routing is deterministic and never silent: ambiguous or unknown names
 * produce errors that list the alternatives.
 */
public final class ModelRouter {

    public record Route(Target target, String modelId, Provider provider) {

        public enum Target {
            LOCAL, PROVIDER
        }
    }

    private final ModelStore store;
    private final ProviderRegistry providers;
    private final LogBuffer log;

    public ModelRouter(ModelStore store, ProviderRegistry providers, LogBuffer log) {
        this.store = store;
        this.providers = providers;
        this.log = log;
    }

    /**
     * Resolution order:
     * 1. exact installed local model id (local models always win);
     * 2. explicit {@code provider:model} or {@code provider/model} prefix;
     * 3. exactly one enabled provider exposing the name — ambiguity is an error.
     */
    public Route resolve(String modelName) {
        if (modelName == null || modelName.isBlank()) {
            throw new IllegalArgumentException("Missing model name");
        }
        if (store.find(modelName) != null) {
            return new Route(Route.Target.LOCAL, modelName, null);
        }
        int colon = modelName.indexOf(':');
        int slash = modelName.indexOf('/');
        int prefixEnd = -1;
        if (colon > 0 && slash > 0) {
            prefixEnd = Math.min(colon, slash);
        } else if (colon > 0) {
            prefixEnd = colon;
        } else if (slash > 0) {
            prefixEnd = slash;
        }
        if (prefixEnd > 0) {
            String providerId = modelName.substring(0, prefixEnd);
            Provider provider = providers.get(providerId);
            if (provider != null) {
                if (!provider.enabled()) {
                    throw new IllegalArgumentException("Provider '" + providerId
                            + "' is disabled; enable it before using model '" + modelName + "'");
                }
                String rest = modelName.substring(prefixEnd + 1);
                if (rest.isBlank()) {
                    throw new IllegalArgumentException("Missing model name after provider prefix in '"
                            + modelName + "'");
                }
                return new Route(Route.Target.PROVIDER, rest, provider);
            }
        }
        List<Provider> matches = new ArrayList<>();
        for (Provider provider : providers.enabled()) {
            try {
                if (providers.models(provider).contains(modelName)) {
                    matches.add(provider);
                }
            } catch (Exception e) {
                log.warn("router", "Could not enumerate models of provider " + provider.id()
                        + ": " + e.getMessage());
            }
        }
        if (matches.size() == 1) {
            return new Route(Route.Target.PROVIDER, modelName, matches.get(0));
        }
        if (matches.size() > 1) {
            List<String> ids = matches.stream().map(Provider::id).sorted().toList();
            throw new IllegalArgumentException("Model '" + modelName + "' is ambiguous across providers "
                    + ids + "; use <provider>:" + modelName + " to disambiguate");
        }
        throw new IllegalArgumentException("Model '" + modelName + "' is not installed and not exposed by any "
                + "configured provider. Install it (enr pull " + modelName + ") or configure a provider "
                + "(enr provider add).");
    }

    /** All routable models: local ids plus {@code provider:model} for providers. */
    public List<String> routableNames() {
        List<String> names = new ArrayList<>();
        for (ModelStore.StoredModel model : store.list()) {
            names.add(model.id());
        }
        for (Provider provider : providers.enabled()) {
            for (String model : providers.models(provider)) {
                names.add(provider.id() + ":" + model);
            }
        }
        return names;
    }
}
