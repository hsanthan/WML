def calc_scale_pred(data):
    """
    Score method.
    """
    try:
        #data = payload['input_data'][0]['values']

        # print(data)
        return data * 2

    except Exception as e:
        return e
