from End_to_End_ML_Pipeline import logger
from End_to_End_ML_Pipeline.config.configuration import ConfigurationManager
from End_to_End_ML_Pipeline.components.evaluation import Evaluation

STAGE_NAME = "Model Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(config = val_config)
        evaluation.evaluation()
        evaluation.save_score()
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e