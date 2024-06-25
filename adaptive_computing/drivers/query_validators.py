def get_query_validator(criterion):
    if criterion == 'absolute_variance':
        return validate_absolute_variance
    if criterion == "percent_variance":
        return validate_percent_variance
    else:
        raise(KeyError)
    
def validate_absolute_variance(value, variance, max_var):
    return variance < max_var

def validate_percent_variance(value, variance, max_percent_var):
    return 100*variance/value < max_percent_var