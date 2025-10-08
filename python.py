import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import io
import docx # Thư viện để đọc file .docx
import numpy as np

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Phương Án Kinh Doanh",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Phương Án Kinh Doanh 📈")
st.caption("Sử dụng Gemini AI để trích xuất dữ liệu, xây dựng dòng tiền và đánh giá hiệu quả dự án.")

# --- Khởi tạo và Cấu hình API Key ---
try:
    API_KEY = st.secrets.get("GEMINI_API_KEY")
    if API_KEY is None:
        st.warning("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng AI.")
        CLIENT = None
    else:
        CLIENT = genai.Client(api_key=API_KEY)
except KeyError:
    st.warning("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng AI.")
    CLIENT = None
except Exception as e:
    st.error(f"Lỗi khởi tạo Gemini Client: {e}")
    CLIENT = None

# --- Hàm đọc nội dung từ file Word (.docx) ---
def read_docx_file(uploaded_file):
    """Đọc nội dung văn bản từ file .docx đã tải lên."""
    document = docx.Document(uploaded_file)
    content = [p.text for p in document.paragraphs]
    return '\n'.join(content)

# --- Hàm lọc thông tin dự án bằng AI (Nhiệm vụ 1) ---
def extract_project_info(doc_content):
    """Sử dụng Gemini để lọc các chỉ số tài chính từ nội dung văn bản."""
    if CLIENT is None:
        raise APIError("Gemini API Client chưa được khởi tạo.")

    prompt = f"""
    Bạn là một chuyên gia phân tích dự án. Nhiệm vụ của bạn là trích xuất 6 thông tin tài chính sau từ văn bản Phương án Kinh doanh:
    1. Vốn đầu tư ban đầu (Initial Investment - I0)
    2. Dòng đời dự án (Project Life - N, đơn vị năm)
    3. Doanh thu thuần hàng năm (Annual Revenue - R)
    4. Chi phí hoạt động hàng năm (Annual Operating Cost - C, không bao gồm Khấu hao và Lãi vay)
    5. Chi phí sử dụng vốn bình quân (WACC, đơn vị %)
    6. Thuế suất (Tax Rate - t, đơn vị %)
    
    Hãy trả lời **CHỈ DUY NHẤT** dưới dạng một chuỗi JSON hợp lệ, với các giá trị là số (hoặc số thập phân, không có đơn vị tiền tệ/năm/%):
    
    Văn bản Phương án Kinh doanh:
    ---
    {doc_content}
    ---
    
    Cấu trúc JSON bắt buộc:
    {{
        "Vốn đầu tư": <số>,
        "Dòng đời dự án": <số>,
        "Doanh thu": <số>,
        "Chi phí": <số>,
        "WACC": <số>,
        "Thuế": <số>
    }}
    """
    
    response = CLIENT.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    # Xử lý chuỗi JSON trả về
    try:
        # Tìm và trích xuất chuỗi JSON thuần túy
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            json_string = json_match.group(0)
            import json
            data = json.loads(json_string)
            return data
        else:
            raise ValueError("Không tìm thấy cấu trúc JSON hợp lệ trong phản hồi của AI.")
            
    except json.JSONDecodeError:
        raise APIError(f"Phản hồi của AI không phải là JSON hợp lệ: {response.text}")

# --- Hàm tính toán chỉ số và xây dựng dòng tiền (Nhiệm vụ 2 & 3) ---
def calculate_project_metrics(data):
    """Xây dựng bảng dòng tiền và tính toán NPV, IRR, PP, DPP."""
    
    # 1. Lấy và Chuẩn hóa Dữ liệu
    I0 = data.get("Vốn đầu tư", 0)
    N = int(data.get("Dòng đời dự án", 0))
    R = data.get("Doanh thu", 0)
    C = data.get("Chi phí", 0)
    WACC = data.get("WACC", 0) / 100.0
    t = data.get("Thuế", 0) / 100.0
    
    if N <= 0:
        raise ValueError("Dòng đời dự án phải lớn hơn 0.")
        
    # Lợi nhuận trước thuế và lãi vay (EBIT)
    EBIT = R - C
    
    # Lợi nhuận sau thuế (EAT)
    EAT = EBIT * (1 - t)
    
    # Dòng tiền thuần hàng năm (Net Cash Flow - NCF)
    # Giả định Khấu hao = 0 theo phương án đơn giản hóa (hoặc đã nằm trong C)
    NCF_yearly = EAT # + Khấu hao
    
    # 2. Xây dựng Bảng Dòng tiền (Nhiệm vụ 2)
    periods = list(range(0, N + 1))
    
    # Khởi tạo Dòng tiền (Chỉ có NCF hoạt động)
    cash_flows = [0] * (N + 1)
    cash_flows[0] = -I0  # Năm 0: Đầu tư ban đầu
    for i in range(1, N + 1):
        cash_flows[i] = NCF_yearly
        
    df_cashflow = pd.DataFrame({
        'Năm': periods,
        'Dòng tiền (CF)': cash_flows,
        'Lợi nhuận sau thuế (EAT)': [0] + [EAT] * N,
        'WACC Chiết khấu (1/(1+WACC)^n)': [1] + [1 / ((1 + WACC)**n) for n in range(1, N + 1)]
    })
    
    # 3. Tính toán các Chỉ số (Nhiệm vụ 3)
    
    # a. NPV (Net Present Value)
    npv_value = np.npv(WACC, cash_flows)
    
    # b. IRR (Internal Rate of Return)
    try:
        irr_value = np.irr(cash_flows)
    except Exception:
        irr_value = np.nan
        
    # c. PP (Payback Period - Thời gian hoàn vốn)
    cumulative_cf = np.cumsum(cash_flows)
    pp_period = np.nan
    
    if (cumulative_cf > 0).any():
        payback_year_index = np.where(cumulative_cf > 0)[0][0]
        if payback_year_index > 0:
            # Năm trước khi hoàn vốn
            prev_cum_cf = cumulative_cf[payback_year_index - 1]
            # Giá trị NCF năm hoàn vốn
            ncft = cash_flows[payback_year_index]
            # Thời gian hoàn vốn = Năm trước + Giá trị còn thiếu / NCF năm đó
            pp_period = (payback_year_index - 1) + (-prev_cum_cf / ncft)
        elif payback_year_index == 0 and cash_flows[0] == 0 and cash_flows[1] > 0:
             pp_period = 0 # Hoàn vốn ngay trong năm 0 (thực tế rất hiếm)
    
    # d. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    discounted_cf = [cf * (1 / ((1 + WACC)**n)) for n, cf in enumerate(cash_flows)]
    cumulative_dcf = np.cumsum(discounted_cf)
    dpp_period = np.nan
    
    if (cumulative_dcf > 0).any():
        dpayback_year_index = np.where(cumulative_dcf > 0)[0][0]
        if dpayback_year_index > 0:
            prev_cum_dcf = cumulative_dcf[dpayback_year_index - 1]
            dcft = discounted_cf[dpayback_year_index]
            dpp_period = (dpayback_year_index - 1) + (-prev_cum_dcf / dcft)
        elif dpayback_year_index == 0 and discounted_cf[0] == 0 and discounted_cf[1] > 0:
            dpp_period = 0

    metrics = {
        "NPV (Tỷ đồng)": npv_value,
        "IRR (%)": irr_value * 100,
        "PP (Năm)": pp_period,
        "DPP (Năm)": dpp_period,
        "WACC (% dùng để so sánh IRR)": WACC * 100,
        "Dòng đời dự án (Năm)": N
    }
    
    return df_cashflow, metrics

# --- Hàm gọi AI Phân tích Chỉ số (Nhiệm vụ 4) ---
def analyze_metrics_with_ai(metrics, cashflow_table):
    """Gửi các chỉ số đánh giá và bảng dòng tiền đến Gemini để phân tích."""
    if CLIENT is None:
        return "Lỗi: Gemini API Client chưa được khởi tạo để phân tích."

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Chỉ tiêu', 'Giá trị'])
    
    prompt = f"""
    Bạn là một chuyên gia thẩm định dự án tài chính. Dựa trên bảng dòng tiền và các chỉ số hiệu quả dự án sau, hãy đưa ra một nhận xét phân tích chuyên sâu về tính khả thi của dự án.
    
    Đánh giá tập trung vào:
    1. **Quyết định Chấp thuận/Từ chối** dự án dựa trên NPV và so sánh IRR với WACC.
    2. **Khả năng sinh lời** (dựa trên NPV và IRR).
    3. **Rủi ro thanh khoản/Thời gian hoàn vốn** (dựa trên PP và DPP).
    
    ---
    Bảng Dòng tiền (Cash Flow):
    {cashflow_table.to_markdown(index=False)}
    
    ---
    Các Chỉ số Hiệu quả Dự án:
    {metrics_df.to_markdown(index=False)}
    
    ---
    
    Yêu cầu: Viết một bài phân tích chuyên nghiệp, đầy đủ và khách quan, bằng Tiếng Việt.
    """
    
    response = CLIENT.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    return response.text

# --- Khởi tạo State và Logic Chính ---
if "project_data" not in st.session_state:
    st.session_state.project_data = {}
if "cashflow_df" not in st.session_state:
    st.session_state.cashflow_df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "doc_content" not in st.session_state:
    st.session_state.doc_content = ""

# ====================================================================
# --- HIỂN THỊ VÀ XỬ LÝ DỮ LIỆU ---
# ====================================================================

uploaded_file = st.file_uploader(
    "1. Tải file Word (.docx) chứa Phương án Kinh doanh",
    type=['docx']
)

if uploaded_file is not None:
    try:
        # Đọc nội dung file Word
        st.session_state.doc_content = read_docx_file(uploaded_file)
        st.info("Đã tải file Word thành công. Sẵn sàng trích xuất dữ liệu.")

        # Nhiệm vụ 1: Lọc dữ liệu
        if st.button("Lọc Dữ liệu Dự án bằng AI 🤖"):
            if CLIENT is None:
                st.error("Không thể thực hiện tác vụ lọc: Thiếu Khóa API Gemini.")
            else:
                with st.spinner('Đang gửi nội dung và chờ AI trích xuất thông tin...'):
                    project_data = extract_project_info(st.session_state.doc_content)
                    st.session_state.project_data = project_data
                    st.success("Trích xuất dữ liệu thành công!")

        if st.session_state.project_data:
            st.subheader("2. Dữ liệu Dự án đã Lọc")
            data_df = pd.DataFrame(list(st.session_state.project_data.items()), columns=['Chỉ tiêu', 'Giá trị'])
            st.dataframe(data_df.style.format({'Giá trị': '{:,.2f}'}), use_container_width=True, hide_index=True)

            # Nhiệm vụ 2 & 3: Xây dựng dòng tiền và tính toán chỉ số
            try:
                st.session_state.cashflow_df, st.session_state.metrics = calculate_project_metrics(st.session_state.project_data)

                # Hiển thị bảng dòng tiền
                st.subheader("3. Bảng Dòng tiền của Dự án")
                st.dataframe(st.session_state.cashflow_df.style.format({
                    'Dòng tiền (CF)': '{:,.0f}',
                    'Lợi nhuận sau thuế (EAT)': '{:,.0f}',
                    'WACC Chiết khấu (1/(1+WACC)^n)': '{:.4f}'
                }), use_container_width=True, hide_index=True)

                # Hiển thị các chỉ số hiệu quả
                st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án")
                col_npv, col_irr, col_pp, col_dpp = st.columns(4)
                
                with col_npv:
                    st.metric(
                        "NPV (Tỷ đồng)", 
                        f"{st.session_state.metrics['NPV (Tỷ đồng)'] / 1000:,.2f} Tỷ", 
                        delta=f"WACC: {st.session_state.metrics['WACC (% dùng để so sánh IRR)']:.2f}%"
                    )
                with col_irr:
                    st.metric(
                        "IRR (%)", 
                        f"{st.session_state.metrics['IRR (%)']:.2f}%" if not np.isnan(st.session_state.metrics['IRR (%)']) else "N/A",
                        delta_color="off"
                    )
                with col_pp:
                    st.metric("PP (Thời gian hoàn vốn)", f"{st.session_state.metrics['PP (Năm)']:.2f} Năm" if not np.isnan(st.session_state.metrics['PP (Năm)']) else "N/A")
                with col_dpp:
                    st.metric("DPP (Hoàn vốn chiết khấu)", f"{st.session_state.metrics['DPP (Năm)']:.2f} Năm" if not np.isnan(st.session_state.metrics['DPP (Năm)']) else "N/A")

                # Nhiệm vụ 5: Yêu cầu AI phân tích (Nhiệm vụ 4)
                if st.button("Yêu cầu AI Phân tích Chỉ số 🧠"):
                    if CLIENT is None:
                         st.error("Không thể thực hiện tác vụ phân tích: Thiếu Khóa API Gemini.")
                    else:
                        with st.spinner('Đang gửi chỉ số và chờ Gemini phân tích...'):
                            ai_analysis = analyze_metrics_with_ai(st.session_state.metrics, st.session_state.cashflow_df)
                            st.markdown("---")
                            st.subheader("5. Kết quả Phân tích từ Gemini AI")
                            st.info(ai_analysis)

            except ValueError as ve:
                st.error(f"Lỗi tính toán: {ve}")
            except Exception as e:
                st.error(f"Lỗi xảy ra trong quá trình tính toán chỉ số: {e}")

    except Exception as e:
        st.error(f"Lỗi khi đọc file Word: {e}")

else:
    st.info("Vui lòng tải lên file Word (.docx) để bắt đầu quá trình phân tích Phương án Kinh doanh.")
    
# ====================================================================
# --- CHỨC NĂNG BỔ SUNG: KHUNG CHAT HỎI ĐÁP VỚI GEMINI ---
# ====================================================================
st.markdown("---")
st.subheader("6. Chat với AI Phân tích 💬")
st.caption("Sau khi tải file và có dữ liệu, bạn có thể hỏi thêm về các chỉ số hoặc dự án.")

if CLIENT is None:
    st.error("Không thể khởi tạo Chatbot: Thiếu Khóa API 'GEMINI_API_KEY'.")
elif st.session_state.metrics is None:
    st.info("Vui lòng hoàn tất quá trình tải file và tính toán chỉ số trước khi chat.")
else:
    try:
        model_name = 'gemini-2.5-flash'
        
        # Tạo dữ liệu tổng hợp để làm ngữ cảnh cho Chatbot
        metrics_df_chat = pd.DataFrame(list(st.session_state.metrics.items()), columns=['Chỉ tiêu', 'Giá trị'])
        
        # System Instruction buộc Gemini trả lời dựa trên dữ liệu đã tạo
        system_instruction = f"""
        Bạn là một chuyên gia thẩm định dự án tài chính, có khả năng trò chuyện. 
        Mọi câu trả lời của bạn phải dựa trên **Bảng dòng tiền** và **Các Chỉ số Hiệu quả Dự án** sau. 
        Hãy trả lời bằng Tiếng Việt.
        
        DỮ LIỆU CƠ SỞ ĐỂ PHÂN TÍCH:
        ---
        1. Chỉ số Hiệu quả:
        {metrics_df_chat.to_markdown(index=False)}
        
        2. Dòng tiền:
        {st.session_state.cashflow_df.to_markdown(index=False)}
        
        ---
        """

        # Khởi tạo chat session nếu chưa có
        if "project_chat_session" not in st.session_state:
            st.session_state.project_chat_session = CLIENT.chats.create(
                model=model_name,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )
            st.session_state.chat_history = []
        
        # 1. Hiển thị lịch sử chat
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 2. Xử lý Input từ người dùng
        if user_prompt := st.chat_input("Hỏi AI: 'Dự án có nên được chấp thuận không?'"):
            
            # Thêm tin nhắn người dùng vào lịch sử và hiển thị
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            # Gửi tin nhắn đến mô hình Gemini và hiển thị phản hồi streaming
            with st.spinner("Đang xử lý..."):
                try:
                    # Gửi tin nhắn đến phiên chat đã khởi tạo
                    response = st.session_state.project_chat_session.send_message(user_prompt, stream=True)
                    
                    full_response = ""
                    with st.chat_message("assistant"):
                        # Dùng st.write_stream để hiển thị phản hồi dần dần
                        full_response = st.write_stream(response)
                    
                    # Lưu phản hồi đầy đủ vào lịch sử chat
                    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                    
                except APIError as e:
                    st.error(f"Lỗi gọi Gemini API trong Chatbot: {e}")
                except Exception as e:
                    st.error(f"Đã xảy ra lỗi không xác định trong Chatbot: {e}")

    except Exception as e:
        st.error(f"Lỗi khởi tạo Chatbot: {e}")
