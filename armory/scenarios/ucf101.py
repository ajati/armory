"""
Classifier evaluation within ARMORY

Scenario Contributor: MITRE Corporation
"""

import logging

import numpy as np
from tqdm import tqdm

from armory.scenarios.base import Scenario
from armory.utils import metrics
from armory.utils.config_loading import load_dataset, load_model, load_attack

logger = logging.getLogger(__name__)


class Ucf101(Scenario):
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """
        model_config = config["model"]
        classifier, preprocessing_fn = load_model(model_config)

        if model_config.get("fit"):
            logger.info(
                f"Fitting model of {model_config['module']}.{model_config['name']}..."
            )
            logger.info(f"Loading training dataset {config['dataset']['name']}...")
            train_epochs = config["model"]["fit_kwargs"]["nb_epochs"]
            batch_size = config["dataset"]["batch_size"]
            train_data_generator = load_dataset(
                config["dataset"],
                epochs=train_epochs,
                split_type="train",
                preprocessing_fn=preprocessing_fn,
            )

            for epoch in range(train_epochs):
                classifier.set_learning_phase(True)

                for _ in tqdm(
                    range(train_data_generator.batches_per_epoch),
                    desc=f"Epoch: {epoch}/{train_epochs}",
                ):
                    x_trains, y_trains = train_data_generator.get_batch()
                    # x_trains consists of one or more videos, each represented as an
                    # ndarray of shape (n_stacks, 3, 16, 112, 112).
                    # To train, randomly sample a batch of stacks
                    x_train = np.zeros(
                        (min(batch_size, len(x_trains)), 3, 16, 112, 112),
                        dtype=np.float32,
                    )
                    for i, xt in enumerate(x_trains):
                        rand_stack = np.random.randint(0, xt.shape[0])
                        x_train[i, ...] = xt[rand_stack, ...]
                    classifier.fit(
                        x_train, y_trains, batch_size=batch_size, nb_epochs=1
                    )

        classifier.set_learning_phase(False)

        # Evaluate ART classifier on test examples
        logger.info(f"Loading testing dataset {config['dataset']['name']}...")
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )

        logger.info("Running inference on benign test examples...")

        metrics_logger = metrics.MetricsLogger.from_config(config["metric"])
        for x_batch, y_batch in tqdm(test_data_generator, desc="Benign"):
            for x, y in zip(x_batch, y_batch):
                y_pred = np.mean(classifier.predict(x), axis=0)
                metrics_logger.update_task(y, y_pred)
        metrics_logger.log_task()

        # Evaluate the ART classifier on adversarial test examples
        logger.info("Generating / testing adversarial examples...")

        attack = load_attack(config["attack"], classifier)
        test_data_generator = load_dataset(
            config["dataset"],
            epochs=1,
            split_type="test",
            preprocessing_fn=preprocessing_fn,
        )
        for x_batch, y_batch in tqdm(test_data_generator, desc="Attack"):
            for x, y in zip(x_batch, y_batch):
                # each x is of shape (n_stack, 3, 16, 112, 112)
                #    n_stack varies
                attack.set_params(batch_size=x.shape[0])
                x_adv = attack.generate(x=x)
                y_pred = np.mean(classifier.predict(x), axis=0)
                metrics_logger.update_task(y, y_pred, adversarial=True)
                metrics_logger.update_perturbation([x], [x_adv])
        metrics_logger.log_task(adversarial=True)
        return metrics_logger.results()