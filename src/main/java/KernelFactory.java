import it.uniroma2.sag.kelp.kernel.Kernel;

/**
 * Returns new instances of kernels for KFold evaluation pipelines.
 */
public interface KernelFactory {
    Kernel getKernel();
}
