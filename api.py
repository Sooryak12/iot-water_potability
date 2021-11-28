import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


app=FastAPI()

class info(BaseModel):
    ph : float
    Hardness: float
    Sulfate : float 
    Turbidity : float 
    Solids: float

        

@app.get('/')
def main():
    return {"message":"working"}

@app.post('/predict')
def deploy(values: info):
    data =values.dict()
    ph =data['ph']
    hard=data['Hardness']
    sulphate=data['Sulfate']
    turb=data['Turbidity']
    solid=data['Solids']
    
    model=pickle.load(open("dtree_model.sav","rb"))
    
    predict=model.predict([[ph,hard,sulphate,turb,solid]])
    
    map={1:"Yes",0:"No"}
    
    out={"Potable":map[predict[0]]}
    
    
    return out


if __name__=="__main__":
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
    
    