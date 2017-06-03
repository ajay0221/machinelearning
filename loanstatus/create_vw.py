import csv

fields = ['Application_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome',
          'CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
numerical_vars = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
target_var = 'Loan_Status'
target_vals_label_mapping = {'Y':'1', 'N':'-1'}

if __name__ == "__main__":
    input_filename = "data/train_data.csv"
    output_filename = "data/train_data.vw"
    with open(input_filename, 'rb') as f:
        with open(output_filename, 'wb') as g:
            reader = csv.reader(f)
            headers = reader.next()
            for row in reader:
                line = target_vals_label_mapping[row[fields.index(target_var)]] +
                for i in range(0,len(fields)):
