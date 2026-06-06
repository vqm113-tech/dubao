import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import io
import urllib3
from scipy.signal import find_peaks
from PIL import Image
from datetime import datetime

# --- CONFIG STREAMLIT (Phải nằm ở đầu tiên) ---
st.set_page_config(
    page_title="Dự báo Hải văn Quảng Trị", 
    page_icon="🌊", 
    layout="wide"
)

# Tắt cảnh báo SSL của urllib3 để tránh gây lỗi môi trường nội bộ
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- TÍCH HỢP TẬP TRUNG CSS ĐỂ KHÔNG LÀM VỠ GIAO DIỆN (DARK MODE CHUẨN CHUYÊN NGHIỆP) ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    div[data-testid="metric-container"] {
        background: #1c2333;
        border: 1px solid #2e3b55;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] label {
        color: #8fa0bc !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #00c6ff !important;
    }
    h1, h2, h3 {
        color: #00c6ff !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. DANH SÁCH TRẠM QUAN TRẮC & OFFSET ĐỘ CAO HẢI ĐỒ ---
LOCATIONS = {
    "Cửa Gianh": {"lat": 17.70, "lon": 106.48, "offset_default": 1.15},
    "Hòn La": {"lat": 17.93, "lon": 106.52, "offset_default": 1.20},
    "Nhật Lệ": {"lat": 17.47, "lon": 106.63, "offset_default": 1.10},
    "Cồn Cỏ": {"lat": 17.15, "lon": 107.34, "offset_default": 0.95},
    "Cửa Việt": {"lat": 16.89, "lon": 107.18, "offset_default": 1.05},
    "Cửa Tùng": {"lat": 17.02, "lon": 107.10, "offset_default": 1.00}
}

# --- 2. HÀM ĐỔI 16 HƯỚNG HẢI VĂN CHUẨN ---
def get_direction_text(degree):
    if degree is None or np.isnan(degree): 
        return "N/A"
    dirs = [
        "B", "BĐB", "ĐB", "ĐĐB",
        "Đ", "ĐĐN", "ĐN", "NĐN",
        "N", "NTN", "TN", "TTN",
        "T", "TTB", "TB", "BTB"
    ]
    idx = int(round(degree / 22.5)) % 16
    return dirs[idx]

# --- 3. HÀM PHÂN CẤP CẢNH BÁO SÓNG NGUY HIỂM ---
def get_wave_warning(h):
    if h >= 4.0:
        return "Biển động rất mạnh (Nguy hiểm)"
    elif h >= 2.0:
        return "Biển động"
    elif h >= 1.25:
        return "Biển động nhẹ"
    else:
        return "Biển tương đối êm"

# --- 4. HÀM HIỆU CHỈNH TỪ FILE THỰC TẾ ---
def get_calibration_from_file(uploaded_file):
    if uploaded_file is not None:
        try:
            df_real = pd.read_excel(uploaded_file)
            col = next((c for c in df_real.columns if c.lower() in ['muc_nuoc', 'mực nước', 'water_level']), None)
            if col:
                return float(df_real[col].mean())
        except:
            return None
    return None

# --- 5. HÀM LẤY DỰ BÁO CÓ CACHE & KIỂM TRA LỖI API CHẶT CHẼ ---
@st.cache_data(ttl=3600)  
def fetch_upgraded_forecast(site_name, days, manual_offset=None):
    if site_name not in LOCATIONS:
        return None
    loc = LOCATIONS[site_name]
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": loc['lat'], 
        "longitude": loc['lon'],
        "hourly": ["wave_height", "wave_period", "wave_direction"],
        "timezone": "Asia/Bangkok",
        "forecast_days": days
    }
    
    try:
        response = requests.get(url, params=params, timeout=30, verify=False)
        response.raise_for_status()
        res = response.json()
        
        if 'hourly' not in res:
            st.error("Dữ liệu trả về từ hệ thống API không hợp lệ (Thiếu trường hourly).")
            return None
            
        df = pd.DataFrame(res['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        
        # --- MÔ HÌNH HÀM TRIỀU ĐA THÀNH PHẦN (M2 + S2 + K1) ---
        t = np.arange(len(df))
        phase_local = (loc['lat'] * 2.5 + loc['lon'] * 1.2) % (2 * np.pi)
        base_level = manual_offset if manual_offset is not None else loc['offset_default']
        
        tide_m2 = 0.5 * np.sin(2 * np.pi * t / 12.42 + phase_local)
        tide_s2 = 0.2 * np.sin(2 * np.pi * t / 12.00 + phase_local * 0.8)
        tide_k1 = 0.15 * np.sin(2 * np.pi * t / 23.93 + phase_local * 1.5)
        
        df['mực nước (m)'] = (base_level + tide_m2 + tide_s2 + tide_k1).round(2)
        df['mực nước MSL (m)'] = (df['mực nước (m)'] - base_level).round(2)
        df['hướng sóng (chữ)'] = df['wave_direction'].apply(get_direction_text)
        df['cảnh báo'] = df['wave_height'].apply(get_wave_warning)
        
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối mạng hoặc máy chủ API từ chối: {e}")
        return None
    except Exception as e:
        st.error(f"Lỗi xử lý hệ thống: {e}")
        return None

# --- 6. GIAO DIỆN HEADER CHÍNH ---
st.title("📡 DỰ BÁO HẢI VĂN QUẢNG TRỊ")
st.caption("Tác giả: Vũ Quang Minh")

# --- 7. CẤU HÌNH THANH SIDEBAR ---
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            st.image("logo.png", width=120)
        except:
            pass
        st.markdown("<br>", unsafe_allow_html=True)

    st.header("📂 Hiệu chỉnh Dữ liệu")
    up_file = st.file_uploader("Nạp file quan trắc thực tế (.xlsx)", type=['xlsx'])
    cal_offset = get_calibration_from_file(up_file)
    if cal_offset is not None:
        st.success(f"Đã hiệu chỉnh thông minh Offset nền: {cal_offset:.2f} m")
        
    st.header("⚙️ CẤU HÌNH TRẠM")
    selected_site = st.selectbox("📍 Chọn vùng biển:", list(LOCATIONS.keys()))
    f_days = st.slider("📅 Số ngày dự báo:", 1, 14, 7)

# Thực thi lấy dữ liệu từ kho Cache dữ liệu
df_fc = fetch_upgraded_forecast(selected_site, f_days, cal_offset)

if df_fc is not None:
    # Lấy danh sách ngày độc nhất
    available_dates = sorted(df_fc['time'].dt.date.unique())
    selected_date = st.sidebar.selectbox("📆 Chọn ngày cần dự báo", available_dates)

    # Lọc dữ liệu theo ngày đã chọn
    df_day = df_fc[df_fc['time'].dt.date == selected_date].copy()

    # Mượt hóa đường triều để tìm đỉnh ròng chính xác
    df_day['tide_smooth'] = df_day['mực nước (m)'].rolling(window=3, center=True).mean().fillna(df_day['mực nước (m)'])
    df_day['tide_msl_smooth'] = df_day['mực nước MSL (m)'].rolling(window=3, center=True).mean().fillna(df_day['mực nước MSL (m)'])
    
    peaks, _ = find_peaks(df_day['tide_smooth'])
    troughs, _ = find_peaks(-df_day['tide_smooth'])

    high_tides = df_day.iloc[peaks]
    low_tides = df_day.iloc[troughs]

    st.subheader("⏱ Phân tích Trạng thái Triều thực tế")
    col_tide1, col_tide2 = st.columns(2)

    with col_tide1:
        if len(high_tides) > 0:
            ht = high_tides.iloc[0]
            st.success(
                f"🌊 **NƯỚC LỚN (ĐỈNH TRIỀU):**\n\n"
                f"• Mực nước Hải đồ: {ht['tide_smooth']:.2f} m\n\n"
                f"• Mực nước MSL: {ht['tide_msl_smooth']:+.2f} m\n\n"
                f"• Thời gian: {ht['time'].strftime('%H:%M')}"
            )
        else:
            st.info("Không có đỉnh triều rõ rệt trong ngày.")

    with col_tide2:
        if len(low_tides) > 0:
            lt = low_tides.iloc[0]
            st.info(
                f"🌊 **NƯỚC RÒNG (CHÂN TRIỀU):**\n\n"
                f"• Mực nước Hải đồ: {lt['tide_smooth']:.2f} m\n\n"
                f"• Mực nước MSL: {lt['tide_msl_smooth']:+.2f} m\n\n"
                f"• Thời gian: {lt['time'].strftime('%H:%M')}"
            )
        else:
            st.info("Không có chân triều rõ rệt trong ngày.")

    # Phân tích xu hướng dòng chảy (Nước dâng / Nước rút)
    df_day['trend'] = df_day['tide_smooth'].diff()
    flood_df = df_day[df_day['trend'] > 0]
    ebb_df = df_day[df_day['trend'] < 0]

    st.markdown("### 🧭 Xu hướng dòng triều trong ngày")
    if not flood_df.empty:
        st.markdown(f"🔺 **Bắt đầu thời kỳ nước lên (Triều dâng):** lúc {flood_df.iloc[0]['time'].strftime('%H:%M')}")
    if not ebb_df.empty:
        st.markdown(f"🔻 **Bắt đầu thời kỳ nước rút (Triều thoái):** lúc {ebb_df.iloc[0]['time'].strftime('%H:%M')}")

    # Hiện chu kỳ hằng số lý thuyết
    with st.expander("📊 Xem Hằng số Chu kỳ triều chuẩn"):
        st.write("• **M2** (Chu kỳ bán nhật triều chính Mặt Trăng): 12.42 giờ")
        st.write("• **S2** (Chu kỳ bán nhật triều chính Mặt Trời): 12.00 giờ")

    # --- PHẦN KẾT LUẬN NHẬN XÉT TỰ ĐỘNG ---
    max_wave = df_day['wave_height'].max()
    if max_wave < 1.0:
        note = "Biển tương đối êm, thuận lợi cho hoạt động hàng hải."
    elif max_wave < 1.25:
        note = "Biển lặng, sóng nhỏ."
    elif max_wave < 2.0:
        note = "Biển động nhẹ, các tàu thuyền nhỏ cần lưu ý."
    elif max_wave < 4.0:
        note = "Biển động, sóng lớn nguy hiểm."
    else:
        note = "Biển động rất mạnh! Cảnh báo nguy hiểm cấp độ lớn cho toàn bộ phương tiện."

    st.info(f"📋 **Nhận xét tổng quan ngày {selected_date.strftime('%d/%m/%Y')}:** {note}")

    # --- PHẦN 1: THỐNG KÊ CỰC TRỊ TRONG NGÀY (ĐÃ BỔ SUNG MSL) ---
    st.subheader(f"📊 Thống kê Cực trị ngày {selected_date.strftime('%d/%m/%Y')}")

    # Chia lưới hiển thị các chỉ số cốt lõi trong ngày
    c1, c2, c3 = st.columns(3)
    c1.metric("🌊 Sóng lớn nhất", f"{max_wave:.2f} m")
    c2.metric("🌊 Sóng nhỏ nhất", f"{df_day['wave_height'].min():.2f} m")
    c3.metric("⏱ Chu kỳ lớn nhất", f"{df_day['wave_period'].max():.1f} s")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c4, c5, c6, c7 = st.columns(4)
    c4.metric("📊 Mực nước Hải đồ Lớn nhất", f"{df_day['mực nước (m)'].max():.2f} m")
    c5.metric("📊 Mực nước Hải đồ Nhỏ nhất", f"{df_day['mực nước (m)'].min():.2f} m")
    c6.metric("🌐 Mực nước MSL Lớn nhất", f"{df_day['mực nước MSL (m)'].max():+.2f} m")
    c7.metric("🌐 Mực nước MSL Nhỏ nhất", f"{df_day['mực nước MSL (m)'].min():+.2f} m")
    
    # Định dạng mốc thời gian xuất hiện cực trị trong ngày
    wave_max_row = df_day.loc[df_day['wave_height'].idxmax()]
    tide_max_row = df_day.loc[df_day['mực nước (m)'].idxmax()]
    tide_min_row = df_day.loc[df_day['mực nước (m)'].idxmin()]
    msl_max_row = df_day.loc[df_day['mực nước MSL (m)'].idxmax()]
    msl_min_row = df_day.loc[df_day['mực nước MSL (m)'].idxmin()]

    st.write("")
    st.markdown(f"✅ **Đỉnh sóng lớn nhất:** đạt `{wave_max_row['wave_height']:.2f} m` vào lúc **{wave_max_row['time'].strftime('%H:%M')}**")
    st.markdown(f"🔺 **Thời điểm triều cường tối đa (Hải đồ):** đạt `{tide_max_row['mực nước (m)']:.2f} m` vào lúc **{tide_max_row['time'].strftime('%H:%M')}** `(MSL: {tide_max_row['mực nước MSL (m)']:+.2f} m)`")
    st.markdown(f"🔻 **Thời điểm nước ròng tối thiểu (Hải đồ):** đạt `{tide_min_row['mực nước (m)']:.2f} m` vào lúc **{tide_min_row['time'].strftime('%H:%M')}** `(MSL: {tide_min_row['mực nước MSL (m)']:+.2f} m)`")

    # --- PHẦN 2: THỐNG KÊ CỰC TRỊ TOÀN CHU KỲ (ĐÃ BỔ SUNG MSL) ---
    st.markdown("---")
    st.subheader("📊 Số liệu thống kê cực trị trong toàn bộ kỳ dự báo (Toàn kỳ)")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Sóng lớn nhất (Hmax)", f"{df_fc['wave_height'].max():.2f} m")
    m2.metric("Sóng trung bình (Htb)", f"{df_fc['wave_height'].mean():.2f} m")
    m3.metric("Chu kỳ trung bình (Ttb)", f"{df_fc['wave_period'].mean():.1f} s")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    m4, m5, m6, m7 = st.columns(4)
    m4.metric("Mực nước Hải đồ cao nhất", f"{df_fc['mực nước (m)'].max():.2f} m")
    m5.metric("Mực nước Hải đồ thấp nhất", f"{df_fc['mực nước (m)'].min():.2f} m")
    m6.metric("Mực nước MSL cao nhất", f"{df_fc['mực nước MSL (m)'].max():+.2f} m")
    m7.metric("Mực nước MSL thấp nhất", f"{df_fc['mực nước MSL (m)'].min():+.2f} m")
    
    # --- PHẦN 3: ĐỒ THỊ BIẾN TRÌNH SÓNG CHI TIẾT ---
    st.markdown("---")
    st.subheader(f"📈 Biến trình Sóng tổng hợp ngày {selected_date.strftime('%d/%m/%Y')} - {selected_site}")
    
    fig_wave = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_wave.add_trace(go.Scatter(
        x=df_day['time'], y=df_day['wave_height'], name="Chiều cao sóng (m)",
        line=dict(color='#1f77b4', width=3),
        customdata=np.stack((df_day['hướng sóng (chữ)'], df_day['cảnh báo']), axis=-1),
        hovertemplate="<b>Thời gian:</b> %{x|%H:%M}<br><b>Độ cao sóng:</b> %{y} m<br><b>Hướng sóng:</b> %{customdata[0]}<br><b>Trạng thái:</b> %{customdata[1]}<extra></extra>"
    ), secondary_y=False)

    fig_wave.add_trace(go.Scatter(
        x=df_day['time'], y=df_day['wave_period'], name="Chu kỳ sóng (s)",
        line=dict(color='#2ca02c', width=2, dash='dot'),
        hovertemplate="<b>Thời gian:</b> %{x|%H:%M}<br><b>Chu kỳ:</b> %{y} s<extra></extra>"
    ), secondary_y=True)

    fig_wave.update_layout(
        hovermode="x unified", template="plotly_dark", height=450,
        xaxis=dict(showspikes=True, spikemode="across", tickformat="%H:%M"),
        yaxis=dict(title="Chiều cao sóng (m)", color="#1f77b4"),
        yaxis2=dict(title="Chu kỳ sóng (s)", color="#2ca02c"),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right")
    )
    st.plotly_chart(fig_wave, use_container_width=True)

    # --- PHẦN 4: ĐỒ THỊ MỰC NƯỚC THỦY TRIỀU CHI TIẾT ---
    st.markdown("---")
    st.subheader(f"🌊 Biến trình Mực nước Thủy triều so với MSL ngày {selected_date.strftime('%d/%m/%Y')} - {selected_site}")
    
    fig_tide = go.Figure()
    fig_tide.add_trace(go.Scatter(
        x=df_day['time'], y=df_day['tide_msl_smooth'], name="Mực nước MSL",
        line=dict(color='#ef553b', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(239, 85, 59, 0.15)',
        hovertemplate="<b>Thời gian:</b> %{x|%H:%M}<br><b>Mực nước MSL:</b> %{y:+.2f} m<extra></extra>"
    ))
    fig_tide.update_layout(
        hovermode="x unified", template="plotly_dark", height=380,
        xaxis=dict(showspikes=True, spikemode="across", tickformat="%H:%M"),
        yaxis=dict(
            title="Mực nước so với MSL (m)", 
            range=[df_fc['mực nước MSL (m)'].min() - 0.3, df_fc['mực nước MSL (m)'].max() + 0.3]
        )
    )
    st.plotly_chart(fig_tide, use_container_width=True)

    # --- PHẦN 5: BẢNG DỮ LIỆU SỐ LIỆU CHI TIẾT THEO NGÀY ---
    st.markdown("---")
    st.subheader(f"📋 Bảng số liệu chi tiết ngày {selected_date.strftime('%d/%m/%Y')}")
    
    df_table_show = df_day.copy()
    df_table_show['time'] = df_table_show['time'].dt.strftime('%H:%M')
    
    # Chỉ giữ lại và đổi tên các cột cần hiển thị thực tế nghiệp vụ
    keep_cols = {
        'time': 'Mốc giờ', 'wave_height': 'Chiều cao sóng (m)', 'wave_period': 'Chu kỳ (s)',
        'wave_direction': 'Hướng (Độ)', 'mực nước (m)': 'Mực nước Hải đồ (m)', 'mực nước MSL (m)': 'Mực nước MSL (m)',
        'hướng sóng (chữ)': 'Hướng sóng', 'cảnh báo': 'Trạng thái'
    }
    df_table_show = df_table_show[list(keep_cols.keys())].rename(columns=keep_cols)
    st.dataframe(df_table_show, use_container_width=True, hide_index=True)

    # --- PHẦN 6: XUẤT FILE BÁO CÁO EXCEL ĐỊNH DẠNG CHUYÊN NGHIỆP ---
    st.markdown("---")
    output = io.BytesIO()
    
    df_export = df_fc.rename(columns={
        'time': 'Thời gian dự báo', 
        'wave_height': 'Chiều cao sóng (m)',
        'wave_period': 'Chu kỳ sóng (s)', 
        'wave_direction': 'Hướng sóng (Độ)',
        'hướng sóng (chữ)': 'Hướng sóng hải văn', 
        'mực nước (m)': 'Mực nước Hải đồ (m)',
        'mực nước MSL (m)': 'Mực nước chuẩn MSL (m)',
        'cảnh báo': 'Cấp độ cảnh báo thiên tai'
    })
    
    # Loại bỏ các cột phụ trợ nháp trước khi lưu sang file Excel công vụ
    drop_cols = ['tide_smooth', 'tide_msl_smooth', 'trend']
    df_export = df_export.drop(columns=[c for c in drop_cols if c in df_export.columns], errors='ignore')
        
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Báo cáo Hải văn')
        workbook  = writer.book
        worksheet = writer.sheets['Báo cáo Hải văn']
        
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'fg_color': '#1F4E78', 'font_color': 'white', 'border': 1
        })
        for col_num, value in enumerate(df_export.columns.values):
            worksheet.write(0, col_num, value, header_format)
            worksheet.set_column(col_num, col_num, 22)
            
    st.download_button(
        label="📥 Tải Báo cáo Hải văn & Thủy triều Chuẩn hóa (.xlsx)", 
        data=output.getvalue(), 
        file_name=f"Bao_cao_nghiep_vu_{selected_site}_{datetime.now().strftime('%d%m%Y')}.xlsx", 
        mime="application/vnd.ms-excel"
    )