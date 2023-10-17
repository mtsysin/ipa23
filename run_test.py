"""
A file to play around and run necessary tests
"""
import test

dataset_tester = test.test_dataloader.TestBDD100k()
model_tester = test.test_model_output.TestModel()
postprocess_tester = test.test_postprocess.TestPostprocess()
evaluate_tester = test.test_evaluate.TestEvaluate()

if __name__=="__main__":
    dataset_tester.test_dataset_scaling_and_reversion()
    # dataset_tester.test_transform()
    # postprocess_tester.test_postprocess_on_dataset_output()
    # postprocess_tester.test_postprocess_on_simple_pretrained_model()
    # model_tester.test_model_ouptut()
    # model_tester.test_dumb_net()
    # evaluate_tester.test_ciou()
    # model_tester.test_mish()
    pass