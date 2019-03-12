import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
from pathlib import Path
import os
from functools import reduce
import matplotlib.pyplot as plt


def make_numeric(x):
    try:
        return float(re.sub(r'[^\d.]+', '', str(x)))
    except:
        return x


def get_valid_filename(s):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def extract_pga_data():

    url = 'https://www.pgatour.com/stats.html'
    html = urlopen(url)
    soup = BeautifulSoup(html, 'lxml')

    soup_links = soup.find_all('a', attrs={'href': re.compile("^/stats/categories")})

    category_links = {}
    for link in soup_links:
        category_links[link.getText()] = link.get("href")

    for category_name, category_link in category_links.items():
        url = f'https://www.pgatour.com{category_link}'
        html = urlopen(url)

        soup = BeautifulSoup(html, 'lxml')
        soup = soup.find('div', class_="section categories")
        soup_links = soup.find_all(attrs={'href': re.compile("^/stats/stat")})

        years = ['2018', '2017', '2016', '2015', '2014', '2013', '2012',
                 '2011', '2010', '2009', '2008', '2007', '2006', '2005',
                 '2004', '2003', '2002', '2001', '2000']

        for year in years:
            for item in soup_links:
                link = item.get("href")
                position = link.rfind(".")
                subfolder = get_valid_filename(category_name)
                filename = get_valid_filename(item.getText())
                output_dir = Path(f'data/{year}/{subfolder}')

                if os.path.exists(output_dir / f'{filename}.csv'):
                    pass
                else:
                    try:
                        link = f"{link[:position]}.{year}.{link[position+1:]}"

                        df = pd.read_html(f'https://www.pgatour.com{link}')[1]
                        df.columns = [f'{filename}_{column}' for column in df.columns.tolist()]

                        output_dir.mkdir(parents=True, exist_ok=True)

                        df.to_csv(output_dir / f'{filename}.csv')
                    except:
                        pass


def transform_pga_data():

    dataframes_merged = []
    for year in os.listdir('data'):

        dataframes = []
        for category in os.listdir(f'data/{year}'):

            for file in os.listdir(f'data/{year}/{category}'):
                if category not in ['SCORING', 'POINTSRANKINGS', 'MONEYFINISHES'] or file == 'All-Around_Ranking.csv':

                    df = pd.read_csv(f'data/{year}/{category}/{file}').iloc[:, 3:]
                    df = df.rename(columns={df.columns[[0]][0]: 'Player Name'})
                    df = df.drop_duplicates(subset=['Player Name'], keep='first')

                    dataframes.append(df)
                else:
                    pass

        df_merged = reduce(lambda left, right: pd.merge(left, right, on='Player Name', how='outer'), dataframes)
        df_merged['Year'] = int(year)
        dataframes_merged.append(df_merged)

    final_df = reduce(lambda top, bottom: pd.concat([top, bottom], sort=False), dataframes_merged)

    final_df.to_csv('pga_stats.csv', index=False)


def plot_feature_importances(df, threshold=0.7, table=True):
    """
    Plots 15 most important feature and the cumulative importance of feature.
    Prints the number of feature needed to reach threshold cumulative importance.

    Parameters
    --------
    df : dataframe
        Dataframe of feature importances. Columns must be feature and importance
    threshold : float, default = 0.9
        Threshold for prining information about cumulative importances

    Return
    --------
    df : dataframe
        Dataframe ordered by feature importances with a normalized column (sums to 1)
        and a cumulative importance column

    """
    # Sort feature according to importance
    df = df.sort_values('importance', ascending=False).reset_index()

    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    df['cumulative_importance'] = np.cumsum(df['importance_normalized'])

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))

    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()

    # Cumulative importance plot
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(len(df))), df['cumulative_importance'], 'r-')
    plt.xlabel('Number of feature')
    plt.ylabel('Cumulative Importance')
    plt.title('Cumulative Feature Importance')
    plt.show()

    importance_index = np.min(np.where(df['cumulative_importance'] > threshold))
    print('%d feature required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    if table:
        return df
