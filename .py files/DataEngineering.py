import pandas as pd
from data_loader import load_data
import unicodedata

# Customers

def clean_customers(df):
    df['customer_city'] = df['customer_city'].apply(normalize_text)
    df['customer_state'] = df['customer_state'].str.upper().str.strip()
    df['customer_zip_code_prefix'] = df['customer_zip_code_prefix'].astype(str)
    return df

# Order Items

def clean_order_items(df):
    df['shipping_limit_date'] = pd.to_datetime(df['shipping_limit_date'])
    return df


# Orders

def clean_orders(df):

    date_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col])

    # Keep only delivered orders
    df = df[df['order_status'] == 'delivered']

    # Drop rows where delivery date is missing
    df = df.dropna(subset=['order_delivered_customer_date'])

    # Fill small missing
    df['order_approved_at'].fillna(df['order_purchase_timestamp'], inplace=True)
    return df


def clean_sellers(df):
    df['seller_zip_code_prefix'] = df['seller_zip_code_prefix'].astype(str)
    df['seller_city'] = df['seller_city'].apply(normalize_text)
    df['seller_state'] = df['seller_state'].str.upper().str.strip()
    return df
# Geolocation
'''
def clean_geolocation(df):
    df['geolocation_zip_code_prefix'] = df['geolocation_zip_code_prefix'].astype(str)
    df['geolocation_city'] = df['geolocation_city'].apply(normalize_text)
    df['geolocation_state'] = df['geolocation_state'].str.upper().str.strip()
    return df
'''

# Order Payments
'''
def clean_payments(df):
    df['payment_type'] = df['payment_type'].str.lower().str.strip()
    return df
'''
# Products
'''
def clean_products(df):
    df['product_category_name'] = df['product_category_name'].fillna('unknown')

    df['product_name_lenght'].fillna(df['product_name_lenght'].median(), inplace=True)
    df['product_description_lenght'].fillna(df['product_description_lenght'].median(), inplace=True)
    df['product_photos_qty'].fillna(df['product_photos_qty'].median(), inplace=True)

    df.dropna(subset=[
        'product_weight_g',
        'product_length_cm',
        'product_height_cm',
        'product_width_cm'
    ], inplace=True)

    return df
'''
# Sellers



# Categories
'''
def clean_categories(df):
    df['product_category_name'] = df['product_category_name'].str.lower().str.strip()
    df['product_category_name_english'] = df['product_category_name_english'].str.lower().str.strip()
    return df

'''
def normalize_text(text):
    if pd.isnull(text):
        return text
    return unicodedata.normalize('NFKD', str(text)).encode('ascii', errors='ignore').decode('utf-8').lower().strip()

def clean_all_data():
    data = load_data()

    df_customers = clean_customers(data["customers"])
    #df_geolocation = clean_geolocation(data["geolocation"])
    df_items = clean_order_items(data["items"])
    #df_order_pay = clean_payments(data["payments"])
    df_orders = clean_orders(data["orders"])
    #df_products = clean_products(data["products"])
    df_sellers = clean_sellers(data["sellers"])
    #df_category = clean_categories(data["category"])

    return {
        "customers": df_customers,
        #"geolocation": df_geolocation,
        "items": df_items,
        #"payments": df_order_pay,
        "orders": df_orders,
        #"products": df_products,
        "sellers": df_sellers,
        #"category": df_category
    }
