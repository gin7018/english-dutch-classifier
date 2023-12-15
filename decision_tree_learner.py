import json
import common


def construct_decision_tree(df, depth=0, max_depth=None):
    if df.shape[0] <= 5 or (max_depth and depth == max_depth):

        class_a = df[(df[common.CLASS] == common.CLASS_A)].shape[0]
        class_b = df[(df[common.CLASS] == common.CLASS_B)].shape[0]
        if class_a > class_b:
            return common.DecisionNode(current_decision=common.CLASS_A)
        return common.DecisionNode(current_decision=common.CLASS_B)

    best_feature = common.find_best_feature_with_info_gain(df)

    decision = common.DecisionNode(current_decision=best_feature)

    left_child = df[df[best_feature] == True]
    right_child = df[df[best_feature] == False]

    decision.left = construct_decision_tree(left_child, depth + 1, max_depth)
    decision.right = construct_decision_tree(right_child, depth + 1, max_depth)

    return decision


def get_classification(tree, record):
    if tree.current_decision == common.CLASS_A:
        return common.CLASS_A
    elif tree.current_decision == common.CLASS_B:
        return common.CLASS_B

    if record[tree.current_decision]:
        return get_classification(tree.left, record)
    return get_classification(tree.right, record)


def actual_error_rate(tree, df):
    incorrect = 0
    for row in df.iterrows():
        classification = get_classification(tree, row[1])
        if classification != row[1][common.CLASS]:
            incorrect += 1

    return incorrect


def save_tree_to_json(decision_node, out):
    json_string_node = json.dumps(decision_node, default=common.serialize_decisions, indent=2)

    tree_json = {
        "learner_id": "decision_tree",
        "classifier": json_string_node
    }

    with open(out, "w") as f:
        json.dump(tree_json, f, indent=2)


def from_json_to_tree(json_decision_node):
    decision_tree = json.loads(json_decision_node)
    return common.deserialize_decision(decision_tree)
