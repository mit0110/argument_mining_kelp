import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.tree.PartialTreeKernel;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.List;

/**
 * Implements a Partial Tree classifier with trained and tested using KFold
 * validation.
 */
public class PartialTreeClassifier {

    private static class PTKernelFactory implements KernelFactory {
        public Kernel getKernel() {
            return new PartialTreeKernel(0.4f, 0.4f, 1, "tree");
        }
    }

    private static void exploreDataset(SimpleDataset dataset, Logger logger) {
        logger.info("Total size: " + dataset.getNumberOfExamples());
        List<Label> classes = dataset.getClassificationLabels();
        for (Label label : classes) {
            logger.info("Class " + label.toString() + ": "
                    + dataset.getNumberOfPositiveExamples(label));
        }
    }

    public static void main(String[] args) throws Exception {
        Logger logger = LogManager.getRootLogger();
        String trainFilePath = args[0];
        SimpleDataset dataset = new SimpleDataset();
        dataset.populate(trainFilePath);
        SimpleDataset trainDataset = dataset.splitClassDistributionInvariant(
                0.2f)[0];

        logger.info("Dataset populated.");
        exploreDataset(trainDataset, logger);

        ClassificationPipeline pipeline = new ClassificationPipeline();
        PTKernelFactory kernelFactory = new PTKernelFactory();
        List<MulticlassClassificationEvaluator> evaluators =
                pipeline.kFoldEvaluate(trainDataset, kernelFactory, 2, 0.8f);
        float meanF1 = 0;
        for (MulticlassClassificationEvaluator evaluator : evaluators) {
            logger.info("Iteration:");
            logger.info("Mean Accuracy: " + evaluator.getAccuracy());
            for (Label label : dataset.getClassificationLabels()) {
                logger.info("Class " + label.toString());
                evaluator.printCounters(label);
            }
            meanF1 += evaluator.getMeanF1();
            // Evaluator must be computed before calling this.
            logger.info("\n" + evaluator.toString());
        }
        logger.info("Mean F1 over " + evaluators.size() + " iterations: "
            + meanF1 / evaluators.size());
    }
}
