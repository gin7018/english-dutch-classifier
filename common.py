import math

import text_lab

CLASS = "class"
CLASS_A = "en"
CLASS_B = "nl"


class DecisionNode:
    def __init__(self, current_decision=None, left=None, right=None):
        self.current_decision = current_decision
        self.left = left
        self.right = right


def serialize_decisions(node):
    if not node:
        return None

    result = {
        "current_decision": node.current_decision,
        "left": serialize_decisions(node.left),
        "right": serialize_decisions(node.right)
    }
    return result


def deserialize_decision(json_node):
    if json_node["current_decision"] == CLASS_A:
        return DecisionNode(current_decision=CLASS_A)
    elif json_node["current_decision"] == CLASS_B:
        return DecisionNode(current_decision=CLASS_B)

    decision = DecisionNode(current_decision=json_node["current_decision"])
    decision.left = deserialize_decision(json_node["left"])
    decision.right = deserialize_decision(json_node["right"])

    return decision


def entropy(df, use_weights=False):
    if df.shape[0] == 0:
        return 0
    try:
        if use_weights:
            class_a_weights = df[(df[CLASS] == CLASS_A)]["weight"].sum() / df["weight"].sum()
            class_b_weights = df[(df[CLASS] == CLASS_B)]["weight"].sum() / df["weight"].sum()
            return - (class_a_weights * math.log2(class_a_weights)) - (class_b_weights * math.log2(class_b_weights))

        class_a = df[(df[CLASS] == CLASS_A)].shape[0] / df.shape[0]
        class_b = df[(df[CLASS] == CLASS_B)].shape[0] / df.shape[0]
        return - (class_a * math.log2(class_a)) - (class_b * math.log2(class_b))
    except ValueError:
        return 0


def find_information_gain_for_feature(df, feature, use_weights=False):
    # entropy of parent node
    parent_node_entropy = entropy(df, use_weights)

    # left child node entropy
    left_child = df[df[feature] == True]
    left_child_entropy = entropy(left_child, use_weights)

    # right child node entropy
    right_child = df[df[feature] == False]
    right_child_entropy = entropy(right_child, use_weights)

    weighted_entropy_of_child_nodes = (((left_child.shape[0] * left_child_entropy) +
                                        (right_child.shape[0] * right_child_entropy))) / df.shape[0]

    information_gain = parent_node_entropy - weighted_entropy_of_child_nodes
    return information_gain


def find_best_feature_with_info_gain(df, use_weights=False):
    features = text_lab.provide_features()

    best_feature = features[0]
    max_info_gain = find_information_gain_for_feature(df, best_feature, use_weights)

    for feature in features:
        feature_information_gain = find_information_gain_for_feature(df, feature, use_weights)

        # best feature is the feature with the highest info gain in this population
        if max_info_gain < feature_information_gain:
            best_feature = feature
            max_info_gain = feature_information_gain

    return best_feature

