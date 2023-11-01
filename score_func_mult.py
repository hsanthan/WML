from calc_pred import calc_scale_pred


def score(payload):
    """
        Score method.
        """
    try:
        data = payload['input_data'][0]['values']

        # print(data)
        return {
            'predictions': [
                {'values': [calc_scale_pred(data)]}  # , describe(data)]}
            ]
        }
    except Exception as e:
        return {'predictions': [{'values': [repr(e)]}]}
