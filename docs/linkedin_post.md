I built an end-to-end sales analytics project in pandas, and then gave it a dashboard. 📊

I wanted to move past tutorial-style pandas and work on the kind of messy data analysts actually get. So I built a full pipeline:

🧹 Cleaning: 24k order lines with duplicates, mixed date formats, "₹1,299.00" prices stored as text, inconsistent casing and missing values. The pipeline fixes them in a chain of small steps, logs every fix, and cuts memory by ~40% with smarter dtypes.

📈 Analysis, where the real learning was:
• Cohort retention: which signup months actually keep customers coming back
• RFM segmentation: a fifth of customers (Champions) drive ~51% of revenue
• Pareto: 36% of customers generate 80% of revenue
• Anomaly detection: a robust z-score (median/MAD) that found all 3 flash-sale spikes I planted
• Forecasting: trend × seasonality, picking the model by hold-out error
• Market basket: product pairs ranked by lift, not just raw counts

🖥️ The dashboard (Streamlit + Altair): filter by date, region, category and channel, and every KPI, chart and table updates. You can also upload your own CSV.

Three things I took away:
1. Cleaning is most of the work, and the part that quietly decides if your numbers are right.
2. Median/MAD beats mean/std for outlier detection. One big spike shouldn't hide the next one.
3. Write tests for invariants ("monthly revenue must sum to total revenue"). They caught real bugs.

Note: the data is synthetic (I generated it to look like a real e-commerce store), so the insights illustrate the methods rather than a real business. The code works on any CSV with the same columns.

Code, README and tests are on GitHub 👇
🔗 [paste your GitHub link here]

If you work with data, I'd love feedback. What would you add next: margin analysis, a better forecasting model, or a live database connection?

#Python #Pandas #DataAnalytics #DataScience #Streamlit #Portfolio #LearningInPublic
