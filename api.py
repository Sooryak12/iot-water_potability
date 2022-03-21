import pickle
from fastapi import FastAPI
import uvicorn
import numpy as np
import nest_asyncio
nest_asyncio.apply()


app=FastAPI()


@app.get('/')

def index():

    return {'message': "This is the home page of this API. Go to {predict/values}"}


@app.get('/{values}')
def deploy(values : str):

        #print("deploy started")
    
        #print(values)
        list_=values.split('x')
        
        
        
        model=pickle.load(open("dtree_model.sav","rb"))
           
        #print(f" List ::  {list_}")
        predict=model.predict(np.array([int(list_[0]),int(list_[1]),int(list_[2]),int(list_[3]),int(list_[4])]).reshape(1,-1))
        
        map={1:"Potable",0:"Not Potable"}
    
        out=map[predict[0]]
        
        return {"Water" : out}





#if __name__=="__main__":
#   uvicorn.run(app, host='127.0.0.1', port=4000, debug=False)
    
    