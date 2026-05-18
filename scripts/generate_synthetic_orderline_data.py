#!/usr/bin/env python3
"""
Generate fully synthetic public-safe e-commerce order-line data.

"""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

schema = {'order_id': 'transaction_id', 'order_line_id': 'transaction_line_id', 'customer_id': 'customer_key', 'item_id': 'item_key', 'session_id': 'session_key', 'order_date': 'order_timestamp', 'dispatch_date': 'dispatch_timestamp', 'delivery_date': 'delivery_timestamp', 'return_date': 'return_timestamp', 'original_revenue': 'net_revenue', 'realized_revenue': 'realized_revenue', 'return_flag': 'was_returned', 'quantity': 'quantity', 'product_category': 'product_family', 'product_subcategory': 'product_group', 'channel': 'acquisition_channel', 'campaign': 'campaign_group', 'traffic_source': 'traffic_source_group', 'carrier': 'fulfillment_partner', 'shipping_delay': 'delivery_delay_days', 'dispatch_delay': 'dispatch_delay_days', 'price': 'sale_price', 'list_price': 'list_price', 'discount': 'discount_amount', 'tax_rate': 'tax_rate', 'tax_amount': 'tax_amount', 'customer_region': 'customer_region', 'loyalty_tier': 'loyalty_tier'}
families = ['Everyday Apparel', 'Home Comfort', 'Digital Accessories', 'Active Lifestyle', 'Kids Essentials', 'Beauty & Care', 'Outdoor Living', 'Office & Study', 'Pet Lifestyle', 'Seasonal Gifts']
groups = {'Everyday Apparel': ['Basics', 'Footwear', 'Outerwear', 'Fit-Sensitive Items', 'Accessories'], 'Home Comfort': ['Textiles', 'Storage', 'Kitchen', 'Decor', 'Small Furniture'], 'Digital Accessories': ['Chargers', 'Cases', 'Audio', 'Smart Helpers', 'Cables'], 'Active Lifestyle': ['Training Gear', 'Travel', 'Hydration', 'Recovery', 'Sportswear'], 'Kids Essentials': ['Learning', 'Play', 'Sleep', 'Clothing', 'Safety'], 'Beauty & Care': ['Skin', 'Hair', 'Wellness', 'Fragrance', 'Tools'], 'Outdoor Living': ['Garden', 'Picnic', 'Lighting', 'Camping', 'Weather Gear'], 'Office & Study': ['Desk Tools', 'Paper Goods', 'Organization', 'Tech Desk', 'Writing'], 'Pet Lifestyle': ['Feeding', 'Toys', 'Grooming', 'Comfort', 'Travel'], 'Seasonal Gifts': ['Holiday', 'Celebration', 'Bundles', 'Personal Gifts', 'Limited Editions']}
channels = ['direct_web', 'organic_search', 'paid_search', 'social_media', 'email_campaign', 'affiliate_partner', 'marketplace']
campaigns = ['always_on', 'spring_refresh', 'summer_event', 'back_to_routine', 'autumn_savings', 'holiday_peak', 'clearance', 'new_customer_offer']
sources = ['type_in', 'search_engine', 'social_feed', 'newsletter', 'comparison_site', 'partner_site', 'retargeting']
partners = ['ParcelSwift', 'NorthStar Logistics', 'UrbanRoute', 'EcoShip', 'MetroFulfill', 'BoxBridge']
modes = ['standard', 'express', 'pickup_point', 'locker', 'economy']
regions = ['Northlake', 'Eastvale', 'Westport', 'Southridge', 'Central Plains', 'Highland', 'Coastal Belt', 'Metro Core']
segs = ['one_time', 'occasional', 'loyal', 'premium', 'bargain', 'high_return']

def generate(rows=300000, year=2024, seed=202404, outdir='data', sample_csv_rows=50000, write_outputs=True):
    rng=np.random.default_rng(seed); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    # order line counts
    counts=rng.choice([1,2,3,4,5,6], size=int(rows/1.75), p=[.52,.25,.12,.06,.03,.02])
    while counts.sum()<rows:
        counts=np.r_[counts, rng.choice([1,2,3,4,5,6], size=2000, p=[.52,.25,.12,.06,.03,.02])]
    cs=counts.cumsum(); last=np.searchsorted(cs, rows); counts=counts[:last+1]; counts[-1]-=(counts.sum()-rows); counts=counts[counts>0]
    n_orders=len(counts); order_idx=np.arange(n_orders); rep=np.repeat(order_idx, counts); line_no=np.concatenate([np.arange(1,c+1) for c in counts])
    n_customers=max(10000, int(rows/8.5)); n_items=max(6000, int(rows/22))
    # customers
    cust_seg=rng.choice(segs, n_customers, p=[.34,.28,.16,.07,.10,.05])
    weights=pd.Series(cust_seg).map({"one_time":.48,"occasional":1.15,"loyal":3.1,"premium":4.6,"bargain":1.8,"high_return":2.0}).values*rng.lognormal(0,.55,n_customers)
    weights=weights/weights.sum()
    loyalty=[rng.choice({"one_time":["new","bronze"],"occasional":["bronze","silver"],"loyal":["silver","gold"],"premium":["gold","platinum"],"bargain":["bronze","silver"],"high_return":["silver","gold"]}[s]) for s in cust_seg]
    order_cust=rng.choice(np.arange(n_customers), n_orders, p=weights)
    # items
    fam_probs=np.array([.17,.12,.105,.10,.12,.10,.08,.085,.06,.06]); fam_probs/=fam_probs.sum()
    item_fam=rng.choice(families,n_items,p=fam_probs); item_group=np.array([rng.choice(groups[f]) for f in item_fam])
    centers={"Everyday Apparel":30,"Home Comfort":45,"Digital Accessories":24,"Active Lifestyle":38,"Kids Essentials":20,"Beauty & Care":17,"Outdoor Living":58,"Office & Study":15,"Pet Lifestyle":18,"Seasonal Gifts":28}
    base=np.array([rng.lognormal(np.log(centers[f]),.45) for f in item_fam]).clip(3,450)
    pop=1/(np.arange(1,n_items+1)**.62); pop=pop/pop.sum(); pop=rng.permutation(pop)
    item_idx=rng.choice(np.arange(n_items),rows,p=pop)
    # dates
    mp=np.array([.067,.061,.071,.074,.078,.075,.072,.081,.086,.092,.122,.121]); mp/=mp.sum()
    omonth=rng.choice(np.arange(1,13),n_orders,p=mp)
    days=np.array([rng.integers(1,pd.Period(f"{year}-{m:02d}").days_in_month+1) for m in omonth])
    hourp=np.array([.020,.012,.008,.006,.006,.010,.025,.045,.055,.060,.060,.058,.055,.058,.062,.066,.070,.074,.075,.070,.060,.045,.030,.020]); hourp/=hourp.sum()
    dates=pd.to_datetime({"year":year,"month":omonth,"day":days,"hour":rng.choice(np.arange(24),n_orders,p=hourp),"minute":rng.integers(0,60,n_orders),"second":rng.integers(0,60,n_orders)})
    # month dependent channels/campaigns
    base_ch=np.array([.24,.18,.15,.12,.12,.08,.11])
    ch=[]
    camp=[]
    for m in omonth:
        p=base_ch.copy()
        if m in [11,12]: p+=np.array([-.02,-.01,.04,.03,.02,0,-.06])
        elif m in [3,6,8,10]: p+=np.array([-.01,0,.02,.01,.01,0,-.03])
        elif m in [1,2]: p+=np.array([.02,0,-.02,-.01,0,0,.01])
        p=np.clip(p,.02,None); p=p/p.sum(); ch.append(rng.choice(channels,p=p))
        probs=np.array([.20,.03,.03,.03,.05,.45,.15,.06]) if m in [11,12] else (np.array([.28,.12,.12,.10,.10,.08,.12,.08]) if m in [3,6,8,10] else np.array([.48,.06,.06,.07,.07,.06,.10,.10]))
        camp.append(rng.choice(campaigns,p=probs/probs.sum()))
    ch=np.array(ch); camp=np.array(camp)
    mode=rng.choice(modes,n_orders,p=[.46,.15,.16,.08,.15]); partner=rng.choice(partners,n_orders)
    dispatch=rng.poisson(1.10,n_orders).clip(0,8)
    base_delay=np.select([mode=="express",mode=="standard",mode=="pickup_point",mode=="locker",mode=="economy"],[1,3,4,3,5],3)
    delivery=(base_delay+rng.poisson(1.0,n_orders)).clip(1,15)
    dispatch_ts=dates+pd.to_timedelta(dispatch,unit="D"); delivery_ts=dispatch_ts+pd.to_timedelta(delivery,unit="D")
    df=pd.DataFrame({
        "transaction_id":np.array([f"TXN-{year}-{i:09d}" for i in range(1,n_orders+1)])[rep],
        "transaction_line_id":[f"LINE-{year}-{i:010d}" for i in range(1,rows+1)],
        "line_number":line_no,
        "customer_key":np.array([f"CUST-{i:08d}" for i in range(1,n_customers+1)])[order_cust[rep]],
        "item_key":np.array([f"ITEM-{i:07d}" for i in range(1,n_items+1)])[item_idx],
        "session_key":np.array([f"SESS-{year}-{i:010d}" for i in range(1,n_orders+1)])[rep],
        "order_timestamp":dates[rep],
        "dispatch_timestamp":dispatch_ts[rep],
        "delivery_timestamp":delivery_ts[rep],
        "customer_segment_seed":cust_seg[order_cust[rep]],
        "customer_region":rng.choice(regions,rows),
        "loyalty_tier":np.array(loyalty)[order_cust[rep]],
        "customer_join_month":rng.choice(pd.period_range(f"{year-3}-01",f"{year}-12",freq="M").astype(str),rows),
        "product_family":item_fam[item_idx],
        "product_group":item_group[item_idx],
        "item_variant":rng.choice(["core","seasonal","premium","value","limited"],rows,p=[.55,.15,.10,.15,.05]),
        "size_code_public":rng.choice(["XS","S","M","L","XL","OneSize","Compact","Large"],rows,p=[.06,.13,.22,.18,.10,.21,.06,.04]),
        "color_group":rng.choice(["neutral","blue","green","red","warm","cool","mixed","not_applicable"],rows),
        "assortment_type":rng.choice(["core_range","seasonal_range","limited_run","clearance"],rows,p=[.62,.20,.06,.12]),
        "acquisition_channel":ch[rep],
        "campaign_group":camp[rep],
        "traffic_source_group":rng.choice(sources,rows),
        "fulfillment_partner":partner[rep],
        "delivery_mode":mode[rep],
        "dispatch_delay_days":dispatch[rep],
        "delivery_delay_days":delivery[rep]})
    # family quantity distributions
    qvals=np.array([1,2,3,4,5,6]); qty=np.empty(rows,dtype=np.int16)
    qmap={"Everyday Apparel":[.75,.15,.055,.025,.012,.008],"Home Comfort":[.83,.11,.035,.015,.007,.003],"Digital Accessories":[.65,.20,.08,.035,.02,.015],"Active Lifestyle":[.79,.13,.045,.02,.01,.005],"Kids Essentials":[.66,.20,.08,.035,.015,.010],"Beauty & Care":[.62,.22,.09,.04,.02,.01],"Outdoor Living":[.88,.08,.025,.008,.004,.003],"Office & Study":[.56,.23,.11,.06,.03,.02],"Pet Lifestyle":[.58,.24,.10,.05,.02,.01],"Seasonal Gifts":[.60,.23,.09,.045,.022,.013]}
    for f, probs in qmap.items():
        mask=df.product_family.values==f; p=np.array(probs); qty[mask]=rng.choice(qvals,mask.sum(),p=p/p.sum())
    df["quantity"]=qty
    list_price=base[item_idx]*rng.lognormal(0,.12,rows)
    month_lift=pd.Series(df.order_timestamp.dt.month).map({1:.02,2:.01,3:.04,4:.03,5:.035,6:.05,7:.025,8:.045,9:.04,10:.05,11:.09,12:.075}).values
    camp_disc=pd.Series(df.campaign_group).map({"always_on":0,"spring_refresh":.10,"summer_event":.13,"back_to_routine":.08,"autumn_savings":.11,"holiday_peak":.17,"clearance":.30,"new_customer_offer":.18}).values
    ch_disc=pd.Series(df.acquisition_channel).map({"direct_web":.01,"organic_search":.02,"paid_search":.04,"social_media":.05,"email_campaign":.06,"affiliate_partner":.05,"marketplace":.03}).values
    fam_disc=pd.Series(df.product_family).map({"Everyday Apparel":.05,"Home Comfort":.04,"Digital Accessories":.03,"Active Lifestyle":.04,"Kids Essentials":.06,"Beauty & Care":.05,"Outdoor Living":.03,"Office & Study":.07,"Pet Lifestyle":.06,"Seasonal Gifts":.08}).values
    zero_prob=np.select([df.campaign_group=="always_on",df.campaign_group=="clearance",df.campaign_group=="holiday_peak"],[.45,.05,.08],.18)
    disc=np.where(rng.random(rows)<zero_prob,0,(camp_disc+ch_disc+fam_disc+month_lift+rng.beta(2,12,rows)*.20)).clip(0,.70)
    sale=list_price*(1-disc); discount=(list_price-sale)*qty; net=sale*qty
    tax_rate=rng.choice([0,.05,.07,.19,.21],rows,p=[.04,.08,.18,.55,.15])
    df["list_price"]=np.round(list_price,2); df["sale_price"]=np.round(sale,2); df["discount_amount"]=np.round(discount,2); df["net_revenue"]=np.round(net,2); df["tax_rate"]=tax_rate; df["tax_amount"]=np.round(net*tax_rate,2)
    # returns
    fam_ret=pd.Series(df.product_family).map({"Everyday Apparel":.160,"Home Comfort":.070,"Digital Accessories":.050,"Active Lifestyle":.105,"Kids Essentials":.095,"Beauty & Care":.060,"Outdoor Living":.075,"Office & Study":.035,"Pet Lifestyle":.050,"Seasonal Gifts":.100}).values
    seg_adj=pd.Series(df.customer_segment_seed).map({"one_time":.008,"occasional":0,"loyal":-.020,"premium":-.010,"bargain":.018,"high_return":.115}).values
    ch_adj=pd.Series(df.acquisition_channel).map({"direct_web":-.012,"organic_search":0,"paid_search":.010,"social_media":.030,"email_campaign":-.018,"affiliate_partner":.015,"marketplace":.026}).values
    camp_adj=pd.Series(df.campaign_group).map({"always_on":-.006,"spring_refresh":.003,"summer_event":.006,"back_to_routine":.004,"autumn_savings":.006,"holiday_peak":.018,"clearance":.014,"new_customer_offer":.010}).values
    m_adj=pd.Series(df.order_timestamp.dt.month).map({1:.020,2:-.010,3:-.004,4:-.012,5:-.002,6:.006,7:-.006,8:.003,9:.008,10:.012,11:.026,12:.018}).values
    q_adj=np.select([qty==1,qty==2,qty==3,np.isin(qty,[4,5]),qty>=6],[0,.006,.010,.016,.022],0)
    price_adj=np.clip((sale-40)/500,-.015,.070); delay_adj=np.clip((df.delivery_delay_days.values-4)/90,-.010,.060)
    fit_adj=np.where(df.product_group.astype(str).str.contains("Fit-Sensitive|Footwear|Clothing"),.070,0)
    prob=(fam_ret+seg_adj+ch_adj+camp_adj+m_adj+q_adj+price_adj+delay_adj+fit_adj+rng.normal(0,.006,rows)).clip(.005,.62)
    ret=rng.random(rows)<prob
    returned_qty=np.where(ret,1,0); multi=ret&(qty>1)&(rng.random(rows)<(.25+.05*(qty>=4))); returned_qty[multi]=rng.integers(1,qty[multi]+1)
    returned_rev=sale*returned_qty; realized=np.maximum(0,net-returned_rev)
    df["was_returned"]=ret.astype(int); df["return_reason_group"]=np.where(ret,rng.choice(["size_or_fit","changed_mind","late_delivery","quality_expectation","duplicate_purchase","other"],rows,p=[.30,.22,.12,.18,.07,.11]),"not_returned")
    df["returned_quantity"]=returned_qty; df["returned_revenue"]=np.round(returned_rev,2); df["realized_revenue"]=np.round(realized,2)
    ret_ts=pd.Series(df.delivery_timestamp)+pd.to_timedelta(rng.integers(3,31,rows),unit="D"); df["return_timestamp"]=ret_ts.where(ret,pd.NaT)
    df["order_year"]=df.order_timestamp.dt.year; df["order_month"]=df.order_timestamp.dt.month; df["order_weekday"]=df.order_timestamp.dt.day_name()
    # Missingness
    for col,rate in {"traffic_source_group":.015,"campaign_group":.010,"fulfillment_partner":.005,"customer_region":.004}.items():
        df.loc[rng.random(rows)<rate,col]=pd.NA
    cols=["transaction_id","transaction_line_id","line_number","customer_key","item_key","session_key","order_timestamp","dispatch_timestamp","delivery_timestamp","return_timestamp","product_family","product_group","item_variant","size_code_public","color_group","assortment_type","customer_segment_seed","customer_region","loyalty_tier","customer_join_month","list_price","sale_price","net_revenue","realized_revenue","discount_amount","quantity","tax_rate","tax_amount","was_returned","return_reason_group","returned_quantity","returned_revenue","fulfillment_partner","delivery_mode","dispatch_delay_days","delivery_delay_days","acquisition_channel","campaign_group","traffic_source_group","order_year","order_month","order_weekday"]
    df=df[cols]
    if write_outputs:
        parquet_path = outdir/f"synthetic_orderline_{year}.parquet"
        csv_full_path = outdir/f"synthetic_orderline_{year}.csv.gz"
        csv_sample_path = outdir/f"synthetic_orderline_{year}_sample.csv.gz"
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"Saved Parquet: {parquet_path.resolve()}")
        except Exception as e:
            print("Parquet export skipped because no Parquet engine is available.")
            print("Install pyarrow or fastparquet locally to create Parquet.")
            print("Reason:", repr(e))
        df.to_csv(csv_full_path, index=False, compression="gzip")
        df.sample(min(sample_csv_rows,len(df)),random_state=seed).to_csv(csv_sample_path,index=False,compression="gzip")
        print(f"Saved full CSV.GZ: {csv_full_path.resolve()}")
        print(f"Saved sample CSV.GZ: {csv_sample_path.resolve()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=300000)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--seed", type=int, default=202404)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--sample-csv-rows", type=int, default=50000)
    args = parser.parse_args()
    df = generate(args.rows, args.year, args.seed, Path(args.output_dir), args.sample_csv_rows, True)
    print("Generated rows:", len(df))
    print("Unique transactions:", df["transaction_id"].nunique())
    print("Return rate:", df["was_returned"].mean())

if __name__ == "__main__":
    main()
