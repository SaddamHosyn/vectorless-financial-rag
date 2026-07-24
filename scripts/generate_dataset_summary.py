import os
import pandas as pd
from pathlib import Path

CSV_PATH = Path(__file__).resolve().parent.parent / "scrape" / "data" / "LoanData_Bondora.csv"
OUTPUT_TXT = Path(__file__).resolve().parent.parent / "scrape" / "data" / "policies" / "bondora_loan_dataset_summary.txt"


def generate_summary():
    print(f"Reading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)

    total_rows = len(df)
    total_amount = df["Amount"].sum()
    avg_amount = df["Amount"].mean()
    avg_interest = df["Interest"].mean()
    avg_duration = df["LoanDuration"].mean()

    country_counts = df["Country"].value_counts().to_dict()
    rating_counts = df["Rating"].value_counts().to_dict()
    status_counts = df["Status"].value_counts().to_dict()

    # Breakdown by country
    country_summary = []
    for country, count in country_counts.items():
        sub = df[df["Country"] == country]
        c_amt = sub["Amount"].sum()
        c_avg_int = sub["Interest"].mean()
        country_summary.append(
            f"  - {country}: {count:,} loans | Total Amount: €{c_amt:,.2f} | Avg Interest: {c_avg_int:.2f}%"
        )

    # Breakdown by rating
    rating_summary = []
    for rating, count in rating_counts.items():
        sub = df[df["Rating"] == rating]
        r_avg_amt = sub["Amount"].mean()
        r_avg_int = sub["Interest"].mean()
        rating_summary.append(
            f"  - Rating {rating}: {count:,} loans | Avg Amount: €{r_avg_amt:,.2f} | Avg Interest: {r_avg_int:.2f}%"
        )

    summary_text = f"""BONDORA LOAN DATASET SUMMARY & METRICS DATA

Dataset Overview:
- Source File: LoanData_Bondora.csv
- Total Loan Records: {total_rows:,} loans
- Total Funded Amount: €{total_amount:,.2f}
- Average Loan Amount: €{avg_amount:,.2f}
- Average Interest Rate: {avg_interest:.2f}%
- Average Loan Duration: {avg_duration:.1f} months

Geographic Distribution (Loans by Country):
{chr(10).join(country_summary)}

Credit Rating Distribution:
{chr(10).join(rating_summary)}

Loan Status Categories:
{chr(10).join([f'  - Status {k}: {v:,} loans' for k, v in status_counts.items()])}

Dataset Schema & Key Fields Description:
- LoanId / LoanNumber: Unique identification key for each loan agreement.
- Country: Borrower residency country code (EE = Estonia, FI = Finland, ES = Spain, SK = Slovakia).
- AppliedAmount / Amount: Requested loan amount vs actual issued loan principal in EUR.
- Interest: Annual interest rate applied to the loan (percentage).
- LoanDuration: Contract duration in months (typically 3 to 120 months).
- MonthlyPayment: Agreed monthly repayment installment amount.
- UseOfLoan: Purpose category of the loan (e.g. debt consolidation, home improvement, vehicle, business).
- IncomeTotal / DebtToIncome: Borrower financial metrics at application time.
- ProbabilityOfDefault (PD) / ExpectedLoss (EL): Risk management metrics calculated at origination.
- LossGivenDefault (LGD): Estimated proportion of loan loss if default occurs.
- Rating: Internal credit risk rating grade assigned to the borrower (from AA down to HR - High Risk).
- Status: Current status of the loan contract (Repaid, Current, Late / Defaulted).
- Restructured: Boolean flag indicating if payment schedule terms were altered or rescheduled.
"""

    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(f"Dataset summary successfully generated at {OUTPUT_TXT}!")


if __name__ == "__main__":
    generate_summary()
