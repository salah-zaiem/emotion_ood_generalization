#!/usr/bin/env python3
"""Recipe for training an emotion recognition system from speech data only using IEMOCAP.
The system classifies 4 emotions ( anger, happiness, sadness, neutrality) with wav2vec2.

To run this recipe, do the following:
> python train_with_wav2vec2.py hparams/train_with_wav2vec2.yaml --data_folder /path/to/IEMOCAP_full_release

For more wav2vec2/HuBERT results, please see https://arxiv.org/pdf/2111.02735.pdf

Authors
 * Yingzhi WANG 2021
"""

import os
import sys
import speechbrain as sb
import torchaudio.transforms as T
from hyperpyyaml import load_hyperpyyaml
import torch

class EmoIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + emotion classifier.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        outputs = feats.detach()
        # last dim will be used for AdaptativeAVG pool
        embeddings = self.modules.embedding_model(outputs, lens)
        outputs = self.modules.classifier(embeddings)

        outputs = self.hparams.log_softmax(outputs)
        return outputs



    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        emoid, _ = batch.emo_encoded

        loss = self.hparams.compute_cost(predictions, emoid)
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, emoid)

        return loss

    def fit_batch(self, batch):
        """Trains the parameters given a single batch in input"""

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()

        self.optimizer.zero_grad()

        return loss.detach()

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.nll_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
                "error_rate": self.error_metrics.summarize("average"),
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(stats["error_rate"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)


            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints,
            self.checkpointer.save_checkpoint()

        # We also write statistics about test data to stdout and to logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.optimizer = self.hparams.opt_class(self.hparams.model.parameters())

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions. We expect `prepare_mini_librispeech` to have been called before
    this, so that the `train.json`, `valid.json`,  and `valid.json` manifest
    files are available.
    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.
    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """
    resampler44 = T.Resample(44100, 16000, dtype=torch.float)
    resampler = T.Resample(48000, 16000, dtype=torch.float)
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        if sig.ndim ==2:
            sig = torch.mean(sig, axis=1)
        else : 
            print("mono audio")
        if "cafe" in wav : 
            sig =resampler(sig)
        elif "IEMOCAP" in wav or "crema" in wav : 
            sig = sig
        elif "ASVP" in wav : 
            sig = sig
        else : 
            sig=resampler44(sig)


        return sig



    # Initialization of the label encoder. The label encoder assignes to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("emo", "emo_encoded")
    def label_pipeline(emo):
        yield emo
        emo_encoded = label_encoder.encode_label_torch(emo)
        yield emo_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "emo_encoded"],
        )
    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mappinng.

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="emo",
    )

    return datasets


# RECIPE BEGINS!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    from iemocap_prepare import prepare_data  # noqa E402

    # Data preparation, to be run on only one process.
    """
    sb.utils.distributed.run_on_main(
        prepare_data,
        kwargs={
            "data_original": hparams["data_folder"],
            "save_json_train": hparams["train_annotation"],
            "save_json_valid": hparams["valid_annotation"],
            "save_json_test": hparams["test_annotation"],
            "split_ratio": [80, 10, 10],
            "different_speakers": hparams["different_speakers"],
            "test_spk_id": hparams["test_spk_id"],
            "seed": hparams["seed"],
        },
    )
    """
    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    emo_id_brain = EmoIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    """
    emo_id_brain.fit(
        epoch_counter=emo_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
    """
    checkpoints = [ x for x in os.listdir(hparams["save_folder"]) if x[0:2]=="CK"]
    print(checkpoints)
    def order(cp1,cp2) : 
        if cp1=="start" :
            return 1
        day, hour = cp1.split("+")[1:3]
        day2, hour2 = cp2.split("+")[1:3]
        dday1 = int(day.split("-")[2])
        dday2 = int(day2.split("-")[2])
        if dday1 < dday2 : 
            return 0
        elif dday2 < dday1: 
            return 1
        h,m,s= hour.split("-")
        h2,m2,s2 = hour2.split("-")
        if int(h)< int(h2):
            return 0
        elif int(h2) < int(h) : 
            return 1 
        if int(m)< int(m2):
            return 0
        elif int(m2) < int(m) : 
            return 1 
    min_cp = "start"
    taken = set()
    ordered_checkpoints = []
    for j in range(len(checkpoints)):
        min_cp="start"
        for i in range(len(checkpoints)):
            if i not in taken : 
                if order( min_cp,checkpoints[i]) :
                    min_index = i
                    min_cp=checkpoints[i]
        taken.add(min_index)
        ordered_checkpoints.append(min_cp)
    ordered_dict={v:k for k,v in enumerate(ordered_checkpoints)} 
    print(ordered_dict)

    checkpoints_dir =hparams["save_folder"]
    # Load the best checkpoint for evaluation
    checkpoints = emo_id_brain.checkpointer.list_checkpoints()
    print(len(checkpoints))
    for ind, ck in enumerate(checkpoints) : 
        ckpt_trace = str(ck.path).split("/")[-1]
        order = ordered_dict[ckpt_trace]
        if order%20==0 : 

            print(order)
            emo_id_brain.checkpointer.load_checkpoint(ck)
            hparams["train_logger"].log_stats(
                    {"Checkpoint order": order},)
            test_stats = emo_id_brain.evaluate(
                test_set=datasets["test"],
                min_key="error_rate",
                test_loader_kwargs=hparams["dataloader_options"],
            )
            print(test_stats)
