import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixIndexKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.KernelCache;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.ArrayList;
import java.util.List;

/**
 * Pipeline to abstract the training and evaluation of a OneVsAll classifier.
 */
public class ClassificationPipeline {

    private OneVsAllLearning learner = null;
    private MulticlassClassificationEvaluator evaluator = null;
    Logger logger = LogManager.getRootLogger();

    public OneVsAllLearning fit(SimpleDataset dataset, Kernel kernel) {
        List<Label> classes = dataset.getClassificationLabels();

        KernelCache cache = new FixIndexKernelCache(5000);
        kernel.setKernelCache(cache);
        BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
        svmSolver.setKernel(kernel);
        float c = 1.0f;
        svmSolver.setCp(c);
        svmSolver.setCn(c);

        learner = new OneVsAllLearning();
        learner.setBaseAlgorithm(svmSolver);
        learner.setLabels(classes);
        // learn the prediction function
        learner.learn(dataset);
        return learner;
    }

    public MulticlassClassificationEvaluator predict(SimpleDataset dataset) {
        Classifier classifier = learner.getPredictionFunction();
        evaluator = new MulticlassClassificationEvaluator(
                dataset.getClassificationLabels());
        for (Example example : dataset.getExamples()) {
            ClassificationOutput output = classifier.predict(
                    dataset.getNextExample());
            evaluator.addCount(example, output);
        }
        return evaluator;
    }

    public List<MulticlassClassificationEvaluator> kFoldEvaluate(
            SimpleDataset dataset,
            KernelFactory kernelFactory,
            Integer kFolds, Float trainSize) {
        List<MulticlassClassificationEvaluator> evaluators =
                new ArrayList<MulticlassClassificationEvaluator>();
        for (int iteration = 0; iteration < kFolds; iteration++) {
            SimpleDataset[] split = dataset.splitClassDistributionInvariant(
                    trainSize);
            SimpleDataset trainDataset = split[0];
            SimpleDataset testDataset = split[1];
            logger.info("Total size: " + dataset.getNumberOfExamples());
            logger.info("Train size: " + trainDataset.getNumberOfExamples()
                    + " - Test size: " + testDataset.getNumberOfExamples());

            // Train new classifier
            fit(trainDataset, kernelFactory.getKernel());
            // Get evaluations
            predict(testDataset);
            // Get scores
            evaluators.add(evaluator);
        }
        return evaluators;
    }
}
