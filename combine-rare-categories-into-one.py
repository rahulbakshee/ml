# combine rare occuring categories based on threshold %
def combine_rare_cats(df, column, threshold=1, new_category="Other"):
    df[column] = df[column].mask(df[column].map(df[column].value_counts(normalize=True))*100 < threshold, new_category)
    return df
