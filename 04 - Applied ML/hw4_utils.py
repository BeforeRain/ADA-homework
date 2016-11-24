def print_important_features(feature_name, feature_val, top_k):
    print ("Features sorted by their score:")
    sorted_feature = sorted(zip(map(lambda x: round(x, 4), feature_val), feature_name), reverse=True)
    sum_fi = 0
    for f_importance in range(top_k):
        sum_fi += sorted_feature[f_importance][0]
        print("%d. %s (%f)" % (f_importance + 1, sorted_feature[f_importance][1], sorted_feature[f_importance][0]))
    print("The feature importance propotion of the top " + str(top_k) + " : %f" % sum_fi )

def print_unimportant_features(feature_name, feature_val, bottom_k):
    print ("Features sorted by their score:")
    sorted_feature = sorted(zip(map(lambda x: round(x, 4), feature_val), feature_name), reverse=False)
    sum_fi = 0
    for f_importance in range(bottom_k):
        sum_fi += sorted_feature[f_importance][0]
        print("%d. %s (%f)" % (f_importance + 1, sorted_feature[f_importance][1], sorted_feature[f_importance][0]))
    print("The feature importance propotion of the bottom " + str(bottom_k) + " : %f" % sum_fi )
