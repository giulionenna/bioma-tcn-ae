import numpy as np
import pandas as pd
import utilities
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

class DataSwat:    
    def __init__(self,
                 data_path = "./data/swat/SWaT_Dataset_Attack_v0.csv",
                 series_length = 100000,
                 num_anomalies = 10,
                 min_anomaly_distance = 2000,
                 window_length = 100, 
                 window_stride = 5,
                 error_window_length = 128,
                 input_columns = ["value"],
                 scale_method = "StandardScaler", # [None, "MinMaxScaler", "StandardScaler"]
                 training_split = 0.8,
                 ratio = 1
                ):
        
        self.data_path = data_path
        self.error_window_length = error_window_length
        self.series_length = series_length
        self.num_anomalies = num_anomalies
        self.min_anomaly_distance = min_anomaly_distance
        self.window_length = window_length
        self.window_stride = window_stride
        self.input_columns = input_columns
        self.scale_method = scale_method
        self.training_split = training_split
        self.ratio = ratio
        
    def build_data(self, scaler_method=None):
        data = pd.read_csv(self.data_path)
        data = data.rename(columns={"Normal/Attack":"label"})
        data.label[data.label!="Normal"]=1
        data.label[data.label=="Normal"]=0
        data = data.drop(' Timestamp', axis=1)
        #data = data.set_index("Timestamp")
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        if scaler_method == 'Standard':
            scaler=StandardScaler()
        else:
            scaler=MinMaxScaler()
        
        norm_feature = pd.DataFrame(scaler.fit_transform(data.iloc[:,:51]))
        norm_feature = norm_feature.iloc[:int(self.ratio*len(norm_feature)), :int(self.ratio*norm_feature.shape[1])]
        n_sensor = len(norm_feature.columns)
        train_df = norm_feature.iloc[:int(self.training_split*len(norm_feature))]
        train_label = data.label.iloc[:int(self.training_split*len(norm_feature))]
        
        test_df = norm_feature.iloc[int(self.training_split*len(norm_feature)):]
        test_label = data.label.iloc[int(self.training_split*len(norm_feature)):]

        X_train = self.slide_window(train_df, self.window_length, self.window_stride)
        X_test = test_df.values[np.newaxis, :, :] 
        return {'train_X': X_train, 'test_X': X_test, 'test_labels': test_label.values}

    def slide_window(self, df: pd.DataFrame, wl: int, stride: int):
        n_subseq =int(np.floor((df.shape[0]-wl)/stride)+1) #DA VERIFICARE
        X = np.zeros([n_subseq, wl, df.shape[1] ]) # n subsequences X subsequence size X n features
        for i in range(n_subseq):
            X[i, :, :] = df.iloc[i*stride:i*stride+wl, :]
        
        return X
        