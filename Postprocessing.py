from utils import *
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: accuracy
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}

    best_thresholds = []
    max_acc = 0
    equal_prop = 0
    while equal_prop<=1:
        curr_thresholds = []
        for key in categorical_results.keys():
            threshold = 0
            while threshold<=1:
                thresholded_output = apply_threshold(categorical_results[key], threshold)
                curr_prop = get_num_predicted_positives(thresholded_output)/len(thresholded_output)
                if abs(curr_prop-equal_prop)<=epsilon:
                    curr_thresholds.append(threshold)
                    break
                threshold+=0.01

        if len(curr_thresholds)==len(categorical_results.keys()):
            dummy_dict = {}
            for ind, key in enumerate(categorical_results.keys()):
                dummy_dict[key] = apply_threshold(categorical_results[key], curr_thresholds[ind])
                curr_acc = get_total_accuracy(dummy_dict)
                if curr_acc>max_acc:
                    max_acc = curr_acc
                    best_thresholds = curr_thresholds
        equal_prop+=0.01
 
    for ind, key in enumerate(categorical_results.keys()):
        thresholds[key] = best_thresholds[ind]
        demographic_parity_data[key] = apply_threshold(categorical_results[key], best_thresholds[ind])
    # Must complete this function!
    return demographic_parity_data, thresholds

    #return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):

    thresholds = {}
    equal_opportunity_data = {}

    best_thresholds = []
    max_acc = 0
    equal_tpr = 0
    while equal_tpr<=1:
        curr_thresholds = []
        for key in categorical_results.keys():
            threshold = 0
            while threshold<=1:
                thresholded_output = apply_threshold(categorical_results[key], threshold)
                curr_tpr = get_true_positive_rate(thresholded_output)
                if abs(curr_tpr-equal_tpr)<=epsilon:
                    curr_thresholds.append(threshold)
                    break
                threshold+=0.01

        if len(curr_thresholds)==len(categorical_results.keys()):
            dummy_dict = {}
            for ind, key in enumerate(categorical_results.keys()):
                dummy_dict[key] = apply_threshold(categorical_results[key], curr_thresholds[ind])
                curr_acc = get_total_accuracy(dummy_dict)
                if curr_acc>max_acc:
                    max_acc = curr_acc
                    best_thresholds = curr_thresholds
        equal_tpr+=0.01

    for ind, key in enumerate(categorical_results.keys()):
        thresholds[key] = best_thresholds[ind]
        equal_opportunity_data[key] = apply_threshold(categorical_results[key], best_thresholds[ind])
     
    # Must complete this function!
    return equal_opportunity_data, thresholds

    #return None, None

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}

    for key in categorical_results.keys():
        max_correct = 0
        best_threshold = 0
        threshold = 0
        while threshold<=1:

            thresholded_output = apply_threshold(categorical_results[key], threshold)
            #thresholded_output is a list of tuples with tuple containing new_label, old_label
            num_correct = get_num_correct(thresholded_output)

            if num_correct > max_correct:
                max_correct = num_correct
                best_threshold = threshold

                mp_data[key] = thresholded_output
                thresholds[key] = threshold

            threshold+=0.01

    # Must complete this function!
    return mp_data, thresholds

    #return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}

    # Must complete this function!
    best_thresholds = []
    max_acc = 0
    equal_ppv = 0
    while equal_ppv<1:
        curr_thresholds = []
        for key in categorical_results.keys():
            threshold = 0
            while threshold<=1:
                thresholded_output = apply_threshold(categorical_results[key], threshold)
                curr_ppv = get_positive_predictive_value(thresholded_output)
                if abs(curr_ppv-equal_ppv)<=epsilon:
                    curr_thresholds.append(threshold)
                    break
                threshold+=0.01

        if len(curr_thresholds)==len(categorical_results.keys()):
            dummy_dict = {}
            for ind, key in enumerate(categorical_results.keys()):
                dummy_dict[key] = apply_threshold(categorical_results[key], curr_thresholds[ind])
                curr_acc = get_total_accuracy(dummy_dict)
                if curr_acc>max_acc:
                    max_acc = curr_acc
                    best_thresholds = curr_thresholds
        equal_ppv+=0.01

    for ind, key in enumerate(categorical_results.keys()):
        thresholds[key] = best_thresholds[ind]
        predictive_parity_data[key] = apply_threshold(categorical_results[key], best_thresholds[ind])

    return predictive_parity_data, thresholds
    #return None, None

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    
    single_threshold_data = {}
    thresholds = {}

    group_length = []
    for group in categorical_results.keys():
        group_length.append(len(categorical_results[group]))

    output = []
    for key in categorical_results.keys():
        output.extend(categorical_results[key])

    max_correct = 0
    best_threshold = 0
    threshold = 0
    while threshold<=1:

        thresholded_output = apply_threshold(output, threshold)
        #thresholded_output is a list of tuples with tuple containing new_label, old_label
        num_correct = get_num_correct(thresholded_output)

        if num_correct > max_correct:
            max_correct = num_correct
            best_threshold = threshold

        threshold+=0.01

    for key in categorical_results.keys():
        single_threshold_data[key] = apply_threshold(categorical_results[key], best_threshold)
        thresholds[key] = best_threshold

    # Must complete this function!
    return single_threshold_data, thresholds
    #return None, None