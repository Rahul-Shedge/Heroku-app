from flask import Flask, request, render_template
import pickle
import pandas as pd
import datetime

app = Flask(__name__)
Load_model = pickle.load(open("Models/Logistic_Model.pickle",'rb'))


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template('index.html')


@app.route('/method', methods=['POST'])
def predict():
    Values = [x for x in request.form.values()]
    Features = [x for x in request.form.keys()]
    sample = dict(zip(Features, Values))

    cols = ['Closed', 'Closed with explanation', 'Closed with monetary relief',
            'Closed with non-monetary relief', 'Closed with relief',
            'Closed without relief', 'Untimely response', 'Timely response?',
            'Sub_product_isNan', 'Sub_issue_isNan',
            'Consumer_complaint_narrative_isNan', 'Company_public_response_isNan',
            'Tags_isNan', 'Consumer_consent_provided_isNan',
            'Issue_Loan_modification_collection_foreclosure',
            'Issue_Incorrect_information_on_credit_report',
            'Issue_Loan_servicing__payments__escrow_account',
            "Issue_Cont'd_attempts_collect_debt_not_owed",
            'Issue_Account_opening__closing__or_management',
            'Issue_Disclosure_verification_of_debt', 'Issue_Communication_tactics',
            'Issue_Deposits_and_withdrawals',
            'Issue_Application__originator__mortgage_broker',
            'Issue_Billing_disputes', 'Company_Bank of America',
            'Company_Wells Fargo & Company', 'Company_JPMorgan Chase & Co.',
            'Company_Equifax', 'Company_Experian',
            'Company_TransUnion Intermediate Holdings, Inc.', 'Company_Citibank',
            'Company_Ocwen', 'Company_Capital One', 'Company_Nationstar Mortgage',
            'Company_Synchrony Financial', 'Company_U.S. Bancorp',
            'Product_Mortgage', 'Product_Debt collection',
            'Product_Credit reporting', 'Product_Credit card',
            'Product_Bank account or service', 'Product_Consumer Loan',
            'Product_Student loan', 'Submitted via_Web', 'Submitted via_Referral',
            'Submitted via_Phone', 'Submitted via_Postal mail', 'State_CA',
            'State_FL', 'State_TX', 'State_NY', 'State_GA', 'State_NJ', 'State_PA',
            'State_IL', 'State_VA', 'State_MD', 'State_OH', 'State_NC', 'State_MI',
            'State_AZ', 'State_WA', 'State_MA', 'State_CO', 'Days']

    ColList = ['Sub-product', 'Sub-issue', 'Consumer complaint narrative', 'Company public response', 'Tags',
               'Consumer consent provided?']
    binaryCatCol = ['Timely response?']
    feat = ['Company', 'Product', 'Submitted via', 'State']

    ColL = {x: v for x, v in sample.items() if x in ColList}
    Binarycc = {x: v for x, v in sample.items() if x in binaryCatCol}
    Featlist = {x: v for x, v in sample.items() if x in feat}
    Days = {x: v for x, v in sample.items() if x in ["Date sent to company", "Date received"]}

    #temp1 = pd.DataFrame(ColL)

    dic = {}
    for i in cols:
        dic[i] = 0

    for col, value in ColL.items():
        name = col.replace('-', '_').replace('?', '').replace(" ", '_') + '_isNan'
        for i in cols:
            if (i == name) and (None == value):
                dic[i] = int(1)


    for col, value in Binarycc.items():
        if value == "Yes":
            dic[col] = int(1)


    dummies = ['Closed', 'Closed with explanation', 'Closed with monetary relief',
               'Closed with non-monetary relief', 'Closed with relief',
               'Closed without relief', 'Untimely response']

    for col, value in sample.items():
        if col == "Company response to consumer":
            for i in dummies:
                if i == value:
                    dic[i] = int(1)

    DummiesI = ['Issue_Loan_modification_collection_foreclosure',
                'Issue_Incorrect_information_on_credit_report',
                'Issue_Loan_servicing__payments__escrow_account',
                "Issue_Cont'd_attempts_collect_debt_not_owed",
                'Issue_Account_opening__closing__or_management',
                'Issue_Disclosure_verification_of_debt', 'Issue_Communication_tactics',
                'Issue_Deposits_and_withdrawals',
                'Issue_Application__originator__mortgage_broker',
                'Issue_Billing_disputes']

    for col, value in sample.items():
        if col == "Issue":
            for i in DummiesI:
                if i == col + "_" + value.replace(',', '_').replace(' ', '_'):
                    dic[i] = int(1)


    for col, value in Featlist.items():
        dic[str(col+'_'+value)] = int(1)


    temp1 = datetime.datetime.strptime(Days["Date sent to company"], '%Y-%m-%d').date()
    temp2 = datetime.datetime.strptime(Days["Date received"], '%Y-%m-%d').date()
    val = int(str(temp2-temp1).rstrip('days, 0:00:00'))
    dic["Days"] = val
    prediction = Load_model.predict([list(dic.values())])
    if prediction[0] == 1:
        Output = "Attention! Consumer is more likely to dispute,you need to give more attention towards the consumer"
    else :
        Output = "Great ! There are very less chances that consumer will dispute. "
    return render_template("result.html", result=Output)


if __name__ == "__main__":
    app.run(debug=True)


