import json
import sys

import pandas as pd

import ada_boost_learner as ada_learner
import decision_tree_learner as dt_learner
import text_lab


def provide_2samples(df):
    total_size = df.shape[0]
    df = df.sample(frac=1, random_state=1111)

    training_data_size = int(0.6 * total_size)
    validation_data_size = (total_size - training_data_size) // 2

    training_data = df.loc[0:training_data_size]
    testing_data = df.loc[(training_data_size + validation_data_size):]

    return training_data, testing_data


def train_decision_tree_model(training_data):
    print("TRAINING DECISION TREE")

    decision_tree = dt_learner.construct_decision_tree(training_data, max_depth=10)
    decision_tree_error_rate = dt_learner.actual_error_rate(decision_tree, training_data) / training_data.shape[0]
    print("decision tree error rate: ", decision_tree_error_rate * 100)

    return decision_tree


def train_ada_boost_model(training_data):
    print("TRAINING ADA BOOST")

    adaboost_learner_stumps = ada_learner.construct_adaboost_stumps(training_data)
    adaboost_error_rate = (ada_learner.actual_adaboost_error_rate(adaboost_learner_stumps, training_data) /
                           training_data.shape[0])
    print("ada boost error rate: ", adaboost_error_rate * 100)
    return adaboost_learner_stumps


def evaluate_decision_tree_learner(decision_tree, validation_data):
    print("VALIDATING DECISION TREE")
    decision_tree_error_rate = dt_learner.actual_error_rate(decision_tree, validation_data) / validation_data.shape[0]
    print("decision tree error rate: ", decision_tree_error_rate * 100)


def evaluate_ada_boost_learner(ada_boost_stumps, validation_data):
    print("VALIDATING ADA BOOST LEARNER")
    adaboost_error_rate = (ada_learner.actual_adaboost_error_rate(ada_boost_stumps, validation_data) /
                           validation_data.shape[0])
    print("ada boost error rate: ", adaboost_error_rate * 100)


def train_official_input(examples_filename, hypothesis_out, learning_type):
    training_data = pd.DataFrame(columns=["sentence", "class"])
    with open(examples_filename, encoding="utf-8") as ex:
        for line in ex:
            classification = line[:2]
            sentence = line[3:]
            training_data.loc[len(training_data)] = {"sentence": sentence, "class": classification}

    text_lab.create_features_for_training(training_data)

    if learning_type == "dt":
        dt = dt_learner.construct_decision_tree(training_data, max_depth=10)
        dt_learner.save_tree_to_json(dt, hypothesis_out)
        evaluate_decision_tree_learner(dt, training_data)
    elif learning_type == "ada":
        ada_stumps = ada_learner.construct_adaboost_stumps(training_data)
        ada_learner.save_learner_stumps_to_json(ada_stumps, hypothesis_out)
        evaluate_ada_boost_learner(ada_stumps, training_data)
    else:
        print(f"no learner for {learning_type}")


def predict_official_input(classifier_json_filename, testing_filename):

    df = pd.DataFrame(columns=["sentence"])
    with open(testing_filename) as f:
        for sent in f:
            df.loc[len(df)] = sent

    text_lab.load_features_for_predicting(df)

    with open(classifier_json_filename, "r") as cl:
        jsoned = json.load(cl)

        if jsoned["learner_id"] == "decision_tree":
            tree = dt_learner.from_json_to_tree(jsoned["classifier"])
            for idx, row in df.iterrows():
                print(dt_learner.get_classification(tree, row))

        elif jsoned["learner_id"] == "ada_boost":
            ada_stumps = ada_learner.from_json_ada_learner(jsoned["classifier"])
            for idx, row in df.iterrows():
                print(ada_learner.get_ada_boost_decision(ada_stumps, row))


def local_testing():
    dataset = pd.read_csv("text/text_material.csv")

    text_lab.create_features_for_training(dataset)
    training_data, testing_data = provide_2samples(dataset)

    dt = train_decision_tree_model(dataset)
    print()
    evaluate_decision_tree_learner(dt, testing_data)

    ada = train_ada_boost_model(training_data)
    print()
    evaluate_ada_boost_learner(ada, testing_data)

    print("SAVING BEST TREE")
    dt_learner.save_tree_to_json(dt, "best.model")


def main():

    if sys.argv[1] == "train":
        examples_filename = sys.argv[2]
        hypothesis_out = sys.argv[3]
        learning_type = sys.argv[4]
        train_official_input(examples_filename, hypothesis_out, learning_type)
    elif sys.argv[1] == "predict":
        classifier_filename = sys.argv[2]
        validation_filename = sys.argv[3]
        predict_official_input(classifier_filename, validation_filename)


if __name__ == '__main__':
    main()
