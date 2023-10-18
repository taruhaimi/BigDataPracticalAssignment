import pandas as pd
import matplotlib.pyplot as plt

def read_file_to_df(filename):
    data = pd.read_json(filename, typ='series')
    value = []
    key = []
    for j in list(range(0, data.size)):
        if list(data[j].keys())[0] != 'points':
            key.append(list(data[j].keys())[0])
            value.append(list(data[j].items())[0][1])
            dictionary = dict(zip(key, value))
       

    if list(data[j].keys())[0] == 'points':
        try:
            start = list(list(list(data[data.size-1].items()))[0][1][0][0].items())[0][1][0]
            dictionary['start_lat'] = list(start[0].items())[0][1]
            dictionary['start_long'] = list(start[1].items())[0][1]
            dictionary['end_lat'] = list(start[0].items())[0][1]
            dictionary['end_long'] = list(start[1].items())[0][1]
        except:
            str = "no detailed data"
            # print('No detailed data recorded')
            
        
    df = pd.DataFrame(dictionary, index = [0])

    return df

def plot_weekday_data(data):
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ax = data.plot(x='weekday', kind='bar', stacked=True,
        title='Weekday vs. Sports')
    ax.legend(title='Sports',loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xticklabels(weekdays)
    plt.show()

def plot_season_data(data):
    seasons = ["Winter","Spring","Summer","Autumn"]
    ax = data.plot(x='season', kind='bar', stacked=True,
        title='Season vs. Sports')
    ax.legend(title='Sports',loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xticklabels(seasons)
    plt.show()