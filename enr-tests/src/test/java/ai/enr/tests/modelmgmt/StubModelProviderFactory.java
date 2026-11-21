package ai.enr.tests.modelmgmt;

import ai.enr.manager.models.ModelProvider;
import ai.enr.manager.models.ModelProviderConfig;
import ai.enr.manager.models.ModelProviderFactory;

/** Test factory: registers the stub provider under id {@code stub} (and any id). */
public final class StubModelProviderFactory implements ModelProviderFactory {

    @Override
    public String id() {
        return "stub";
    }

    @Override
    public ModelProvider create(ModelProviderConfig config) {
        return StubModelProvider.instance(config.providerId());
    }
}
