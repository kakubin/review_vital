import pandas as pd


def resize_data_frame(df, size, ratio, threshold):
    positive = round(size * ratio)
    negative = round(size * (1 - ratio))
    df1 = fetch_border_data(df, positive, threshold, True)
    df2 = fetch_border_data(df, negative, threshold, False)
    return pd.concat([df1, df2])


def fetch_border_data(df, size, threshold, positive=True):
    if positive:
        result = df[df['vote'] >= threshold]
    else:
        result = df[df['vote'] < threshold]

    return result.sample(size)


class ReviewData:
    @classmethod
    def fetch(_cls, *data_labels, **options):
        results = []

        if 'size' in options:
            options['size'] = options['size'] // len(data_labels)

        for data_label in data_labels:
            filepath = './data/{}.csv.gz'.format(data_label)
            df = pd.read_csv(filepath)
            if options.keys() >= {'ratio', 'size', 'threshold'}:
                df = resize_data_frame(df, **options).sample(frac=1)
            results.append(df)

        return pd.concat(results)
