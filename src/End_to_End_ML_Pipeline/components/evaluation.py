import tensorflow as tf
from pathlib import Path
from End_to_End_ML_Pipeline.utils.common import save_json
from End_to_End_ML_Pipeline.entity.config_entity import EvaluationConfig


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split = 0.30
        )

        dataflow_kwargs = dict(
            target_size = self.config.param_image_size[:-1],
            batch_size = self.config.param_batch_size,
            interpolation = "bilinear"
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self._valid_generator = valid_datagenerator.flow_from_directory(
            directory = self.config.train_data,
            subset = "validation",
            shuffle = False,
            **dataflow_kwargs
        )
        
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        model = tf.keras.models.load_model(path)
        return model
        
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.scores = self.model.evaluate(self._valid_generator)
            
    def save_score(self):
        scores = {
            "loss": self.scores[0],
            "accuracy": self.scores[1]
        }
        save_json(path = Path("scores.json"), data = scores)