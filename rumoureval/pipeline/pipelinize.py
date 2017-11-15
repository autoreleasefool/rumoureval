"""Transform a function to fit in a Pipeline."""


from sklearn.preprocessing import FunctionTransformer


def pipelinize(function, active=True):
    """Transform a function to fit in a Pipeline.
    Source: https://ryan-cranfill.github.io/sentiment-pipeline-sklearn-3/

    :param function:
        the function to transform
    :type function:
        `function`
    :param active:
        True to enable the function, False to skip it
    :type active:
        `bool`
    :rtype:
        :class:`Transformer`
    """
    def list_comprehend_a_function(list_or_series, active=True):
        """List comprehension to apply a function to a list and return results as a list."""
        if active:
            return [function(i) for i in list_or_series]
        return list_or_series

    return FunctionTransformer(
        list_comprehend_a_function,
        validate=False,
        kw_args={'active': active}
    )
