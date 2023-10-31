def score(payload):
        """
        Score method.
        """
        try:
            data = payload['input_data'][0]['values']
            # print(data)
            return {
                'predictions': [
                    {'values': [(data)*2]}  # , describe(data)]}
                ]
            }
        except Exception as e:
            return {'predictions': [{'values': [repr(e)]}]}

