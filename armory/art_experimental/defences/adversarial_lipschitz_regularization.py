"""
This module calls the module that implements adversarial lipschitz regularization.
"""

import logging

from art.defences.trainer.trainer import Trainer
from art.defences.trainer.adversarial_trainer import AdversarialTrainer
from art.attacks.evasion.projected_gradient_descent import ProjectedGradientDescent

logger = logging.getLogger(__name__)

from ALR.dev.models import ALR
import torch
import pdb
import numpy as np

class AdversarialLipschitzRegularization(Trainer):
    def __init__(self, classifier=None, alr_weight=1.0):
        self.classifier  = classifier 
        self.model = classifier._model._model
        self.alr = ALR()
        self.criterion = classifier._loss
        self.alr_weight = alr_weight
        self.optimizer = classifier._optimizer
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)

    def fit(self, x, y, **kwargs):
        #self.trainer.fit(x, y, **kwargs)
        raise NotImplementedError

    def fit_generator(self, generator, nb_epochs=20, **kwargs):
        for epoch in range(nb_epochs):
            loss_epoch = []
            acc_epoch = []

            for batch_idx in range(int(generator.size / generator.batch_size)):
                inputs, labels = generator.get_batch()

                inputs, labels = torch.from_numpy(inputs).to(self._device), torch.from_numpy(labels).to(self._device)
                self.model.train()

                outputs = self.model(inputs)
                class_loss = self.criterion(outputs, labels)

                if self.alr_weight > 0:
                    input_adv = self.alr.get_adversarial_perturbations(self.model, inputs, labels)
                    output_adv = self.model(input_adv)
                    alr_outputs = self.alr.get_alp(inputs, input_adv, outputs, output_adv, labels)
                    alr_loss = alr_outputs[0].mean()
                    loss = class_loss + self.alr_weight * alr_loss
                else:
                    loss = class_loss

                # Model computations
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # sum training loss
                training_loss = loss.item()

                # training accuracy 
                acc_ = np.mean((torch.argmax(outputs, dim=1) == labels).detach().cpu().numpy())
                loss_epoch.append(loss.item())
                acc_epoch.append(acc_)

                if batch_idx%1==0:
                    message = f"Batch [{batch_idx}] train-loss: {training_loss:.4f}    train-acc: {acc_}   "
                    if self.alr_weight > 0:
                        message += f"alr={alr_loss.item():.4e} "
                    print(message, end="\r")
                    logging.info(message)

            msg_ = f"Epoch {epoch}: loss = {np.mean(loss_epoch)}    acc = {np.mean(acc_epoch)}"
            print(msg_)
            logging.info(msg_) 
            loss_epoch=[]
            acc_epoch=[]

        logging.info("Finished Training")
        torch.save(self.model, "alr_model.pth")


    def get_classifier(self):
        return self.classifier
