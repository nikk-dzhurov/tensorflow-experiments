from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

import stl10
import common
from classifier import Classifier


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    args = common.parse_known_args(argv)
    print("Running with arguments: ", args)

    ds_module = stl10

    ds_module.build_app_flags()

    train_epochs = 45
    train_steps = 1000
    if type(args.steps) is int and 1 <= args.steps <= 10000:
        train_steps = args.steps
    if type(args.epochs) is int and 1 <= args.epochs <= 100:
        train_epochs = args.epochs

    classifier = Classifier(
        model_fn=ds_module.model_fn,
        model_params=ds_module.get_model_params(),
        class_names=ds_module.get_class_names()
    )

    if args.mode == common.TRAIN_EVAL_MODE:
        classifier.train(
            epochs=train_epochs,
            steps=train_steps,
            clean_old_model_data=args.clean,
            eval_after_each_epoch=True,
            load_eval_ds_fn=ds_module.load_eval_dataset,
            load_train_ds_fn=ds_module.load_train_dataset,
        )
    elif args.mode == common.TRAIN_MODE:
        classifier.train(
            epochs=train_epochs,
            steps=train_steps,
            clean_old_model_data=args.clean,
            eval_after_each_epoch=False,
            load_train_ds_fn=ds_module.load_train_dataset,
        )
    elif args.mode == common.EVAL_MODE:
        classifier.eval(
            load_eval_ds_fn=ds_module.load_eval_dataset,
        )
    elif args.mode == common.PREDICT_MODE:
        classifier.predict()
    else:
        print("Model mode \"{}\" is not supported".format(args.mode))
        sys.exit(1)


if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)
