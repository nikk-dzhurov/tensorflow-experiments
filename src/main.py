import sys
import argparse
import tensorflow as tf
from flask import Flask, jsonify, request

import stl10
from classifier import Classifier

EVAL_MODE = "eval"
TRAIN_MODE = "train"
SERVER_MODE = "server"
PREDICT_MODE = "predict"
TRAIN_EVAL_MODE = "train_eval"
PREPARE_DATA_MODE = "prepare_data"


def parse_known_args(argv):
    """Parse application arguments passed through command line"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--clean',
        type=bool,
        default=False,
        help="Remove all model data and start new training"
    )
    parser.add_argument(
        '--mode',
        type=str,
        default=TRAIN_EVAL_MODE,
        help="Model mode"
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=0,
        help="Set training epochs"
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=0,
        help="Set training steps per epoch"
    )
    parser.add_argument(
        '--port',
        type=int,
        default=3000,
        help="Set server port"
    )

    parsed_args, _ = parser.parse_known_args(argv)

    return parsed_args


def main(argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    args = parse_known_args(argv)
    print("Execute the application with arguments: ", args)

    ds_module = stl10

    if args.mode == PREPARE_DATA_MODE:
        ds_module.extend_original_data()
        return 0

    classifier = Classifier(ds_module=stl10)

    if args.mode == TRAIN_EVAL_MODE or args.mode == TRAIN_MODE:
        clean_dir = False
        train_epochs = 3
        train_steps = 20
        if type(args.steps) is int and 1 <= args.steps <= 10000:
            train_steps = args.steps
        if type(args.epochs) is int and 1 <= args.epochs <= 100:
            train_epochs = args.epochs
        if type(args.clean) is bool:
            clean_dir = args.clean

        if args.mode == TRAIN_EVAL_MODE:
            classifier.train(
                steps=train_steps,
                epochs=train_epochs,
                clean_old_model_data=clean_dir,
                eval_after_each_epoch=True,
            )
        else:
            classifier.train(
                steps=train_steps,
                epochs=train_epochs,
                clean_old_model_data=clean_dir,
                eval_after_each_epoch=False,
            )
    elif args.mode == EVAL_MODE:
        classifier.eval()
    elif args.mode == PREDICT_MODE:
        if type(args.image_file) is str and args.image_file != "":
            classifier.predict_image_label(image_location=args.image_file)
        else:
            classifier.predict()
    elif args.mode == SERVER_MODE:
        serv = Flask(__name__)

        @serv.route('/predict', methods=['POST'])
        def predict():
            try:
                result = classifier.predict_image_label_bytes(request.data)
                return jsonify(result), 200
            except Exception:
                return jsonify({
                    "error": "Invalid image"
                }), 400

        @serv.route('/ping', methods=['GET'])
        def ping():
            return "OK"

        port = 3000
        if 1024 < args.port < 32768:
            port = args.port

        serv.run(host='0.0.0.0', port=port)
    else:
        print("Model mode \"{}\" is not supported".format(args.mode))
        sys.exit(1)


if __name__ == "__main__":
    tf.compat.v1.app.run(main=main, argv=sys.argv)
