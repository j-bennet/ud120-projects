#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    
    ### your code goes here
    errors = []
    for i in range(len(predictions)):
        err = abs(predictions[i] - net_worths[i])
        errors.append((err, i))

    n_exclude = int((0.0 + len(predictions)) / 10)
    errors = sorted(errors, reverse=True)

    for i in range(n_exclude, len(errors)):
        err, index = errors[i]
        cleaned_data.append((ages[index], net_worths[index], err))

    return cleaned_data

