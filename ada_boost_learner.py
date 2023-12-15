import json
import math

import numpy as np

import common


def find_best_stump(df):

    best_feature = common.find_best_feature_with_info_gain(df, use_weights=True)

    decision = common.DecisionNode(current_decision=best_feature)

    left_child = df[df[best_feature] == True]
    right_child = df[df[best_feature] == False]

    def get_decision(data):
        class_a = data[(data[common.CLASS] == common.CLASS_A)].shape[0]
        class_b = data[(data[common.CLASS] == common.CLASS_B)].shape[0]
        if class_a >= class_b:
            return common.DecisionNode(current_decision=common.CLASS_A)
        return common.DecisionNode(current_decision=common.CLASS_B)

    decision.left = get_decision(left_child)
    decision.right = get_decision(right_child)

    return decision


def update_weights(stump, df, amount_of_say):
    for index, record in df.iterrows():
        classification = get_classification(stump, record)
        if classification != record[common.CLASS]:
            df.loc[index, "weight"] = record["weight"] * math.exp(amount_of_say)  # misclassified record
        else:
            df.loc[index, "weight"] = record["weight"] * math.exp(-amount_of_say)  # correctly classified


def generated_new_data_set(df):
    if "level_0" in df.columns:
        df.drop("level_0", axis=1, inplace=True)

    df.reset_index(inplace=True)
    ii = df.index.tolist()

    sample_weights = np.random.choice(ii, len(df), p=df["weight"])
    copy = df.iloc[sample_weights]

    if len(copy) != len(df):
        raise ValueError(f"DIFF LENGTHS!!!!! COPY LEN: {len(copy)}, DF LEN: {len(df)}")

    return copy


def construct_adaboost_stumps(df):
    stump_authority = {}
    max_number_of_weak_learners = 50
    weak_learners = 0

    while weak_learners < max_number_of_weak_learners:

        decision_stump = find_best_stump(df)

        total_error = error_rate_with_weights(decision_stump, df)

        if total_error == 0:  # we don't want the log to freak out if te=0
            # print(f"stopping at {weak_learners}: {total_error}")
            break

        stump_amount_of_say = 0.5 * math.log((1 - total_error) / total_error)
        stump_authority[decision_stump] = stump_amount_of_say
        # print(f"learner: {weak_learners} say: {stump_amount_of_say}")

        update_weights(decision_stump, df, stump_amount_of_say)

        cc = df.copy()
        df["weight"] = (df["weight"]) / cc["weight"].sum()

        # print(f"learner {weak_learners} DONE.\n")

        weak_learners += 1
    return stump_authority


def get_classification(decision_stump, record):
    if record[decision_stump.current_decision]:
        return decision_stump.left.current_decision
    return decision_stump.right.current_decision


def get_ada_boost_decision(decision_stumps, record):
    class_a_voter_power = 0
    class_b_voter_power = 0

    for decision_stump, authority in decision_stumps.items():
        classification = get_classification(decision_stump, record)

        if classification == common.CLASS_A:
            class_a_voter_power += authority
        elif classification == common.CLASS_B:
            class_b_voter_power += authority

    if class_a_voter_power > class_b_voter_power:
        return common.CLASS_A
    return common.CLASS_B


def error_rate_with_weights(stump, df):
    incorrect = 0

    for row in df.iterrows():
        classification = get_classification(stump, row[1])
        if classification != row[1][common.CLASS]:
            incorrect += row[1]["weight"]

    return incorrect


def actual_adaboost_error_rate(decision_stumps, df):
    incorrect = 0
    for row in df.iterrows():
        classification = get_ada_boost_decision(decision_stumps, row[1])
        if classification != row[1][common.CLASS]:
            incorrect += 1

    return incorrect


def save_learner_stumps_to_json(decision_stumps, out):
    json_stumps = []

    for decision_stump, amount_of_say in decision_stumps.items():
        stump = {
            "stump": common.serialize_decisions(decision_stump),
            "amount_of_say": amount_of_say
        }

        json_stumps.append(stump)

    json_stumps = json.dumps(json_stumps)

    ada_json = {
        "learner_id": "ada_boost",
        "classifier": json_stumps
    }

    with open(out, "w") as f:
        json.dump(ada_json, f)


def from_json_ada_learner(json_stumps):
    decision_tree_stumps = json.loads(json_stumps)
    decision_stumps = {}

    for stump in decision_tree_stumps:
        decision_stump = common.deserialize_decision(stump["stump"])
        decision_stumps[decision_stump] = stump["amount_of_say"]

    return decision_stumps
