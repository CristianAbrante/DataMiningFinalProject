import pandas as pd



#Main
if __name__ == "__main__":
	#Read df
	df=pd.read_csv('msdata.csv',index_col='id')
	
	#Shuffle data set, same each run
	df=df.sample(frac=1,random_state=1)
	classes=df.pop('class')
	
	#Check null values
	for column in df.columns:
		if(df[column].isnull().any()==True):
			print(column)
	
	#Check data types
	for column in df.columns:
		if(df[column].dtype!=float):
			print(column)
			
	#Seems everything ok
	print(df.describe())	
    
