import pandas as pd
import joblib

model = joblib.load('house_price_model.pkl')

df = pd.read_csv('PuneR.xls')
categorical_features = [
    'furnishing', 'avalable_for', 'address', 'facing',
    'floor_type', 'gate_community', 'corner_pro', 'propertyage'
]
numeric_features = ['bedroom', 'bathrooms', 'area', 'floor_number',
                   'parking', 'aggDur', 'noticeDur', 'lightbill',
                   'powerbackup', 'no_room', 'pooja_room',
                   'study_room', 'others', 'servant_room',
                   'store_room', 'brok_amt',
                   'deposit_amt', 'mnt_amt']
categorical_features.append('maintenance_amt')

def predict_rent(locality, bhk):
    input_data = pd.DataFrame({
        'bedroom': [bhk],
        'bathrooms': [2],
        'area': [0],
        'furnishing': ['Semifurnished'],
        'avalable_for': ['All'],
        'address': [f"{locality}, Pune, Maharashtra"],
        'floor_number': [1],
        'facing': ['No Direction'],
        'floor_type': ['Not provided'],
        'gate_community': ['No'],
        'corner_pro': ['No'],
        'parking': [1],
        'wheelchairadption': [0],
        'petfacility': [0],
        'aggDur': [0],
        'noticeDur': [0],
        'lightbill': [1],
        'powerbackup': [0],
        'propertyage': ['1 to 5 Year Old'],
        'no_room': [1],
        'pooja_room': [0],
        'study_room': [0],
        'others': [0],
        'servant_room': [0],
        'store_room': [0],
        'maintenance_amt': [0],
        'brok_amt': [0],
        'deposit_amt': [0],
        'mnt_amt': [0]
    })
    for feature in categorical_features:
        input_data[feature] = input_data[feature].astype(str)

    predicted_rent = model.predict(input_data)

    matching_entries = df[(df['address'].str.contains(locality, case=False)) &
                          (df['bedroom'] == bhk)]

    print(f"Predicted Rent for {bhk} BHK in {locality}: ₹{predicted_rent[0]:.2f}")
    print("\nMatching Entries:\n", matching_entries)

user_locality = input("Enter locality: ")
user_bhk = int(input("Enter bhk: "))
predict_rent(user_locality, user_bhk)
