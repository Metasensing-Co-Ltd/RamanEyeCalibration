import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from csv_processor import StreamlitRamanSpectrumProcessor

st.set_page_config(
    page_title="Interactive Raman Spectrum Calibration",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Interactive Raman Spectrum Peak Adjustment")
st.markdown("Calibrate your Raman spectrometer by adjusting peak positions and wavenumber values")

# Initialize processor
processor = StreamlitRamanSpectrumProcessor()

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Laser wavelength input
    laser_wavelength = st.number_input(
        "Laser Wavelength (nm)", 
        value=532, 
        min_value=200, 
        max_value=2000,
        step=1
    )
    processor.laser_wavelength = laser_wavelength
    
    st.divider()
    
    # File upload
    st.header("📁 Data Input")
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Upload your Raman spectrum CSV file"
    )
    
    # Show peak detection parameters only when file is uploaded
    if uploaded_file is not None:
        # Peak detection parameters
        st.header("🔍 Peak Detection")
        prominence = st.slider("Peak Prominence", 10, 200, 50)
        distance = st.slider("Min Peak Distance", 5, 50, 20)
    
    # Show coefficient verification only when no file is uploaded
    if uploaded_file is None:
        st.divider()
        
        # Coefficient verification section
        st.header("🔍 Coefficient Verification")
        st.markdown("Load saved polynomial coefficients to verify calibration")
        
        # Input format selection
        coeff_format = st.selectbox(
            "Coefficient Format",
            ["Format 1 (B_0 = value)", "Format 2 ('b_coeff': [...])", "Format 3 ([...])"],
            help="Select the format of your saved coefficients"
        )
        
        # Text area for coefficient input
        coeff_input = st.text_area(
            "Paste Coefficients Here",
            height=150,
            placeholder="Paste your polynomial coefficients here...\n\nFormat 1 example:\nB_0 = 8.0964272024e+03\nB_1 = -2.4747377411e+01\n...\n\nFormat 2 example:\n'b_coeff': ['+1.0812918E+03', '-8.7966558E-02', ...]\n\nFormat 3 example:\n['+1.0812918E+03', '-8.7966558E-02', ...]"
        )
        
        # Parse coefficients button
        if st.button("📊 Verify Coefficients", use_container_width=True):
            if coeff_input.strip():
                try:
                    coeffs_parsed = None
                    
                    if "Format 1" in coeff_format:
                        # Parse Format 1: B_0 = value format
                        lines = coeff_input.strip().split('\n')
                        coeffs_parsed = []
                        for line in lines:
                            if '=' in line and 'B_' in line:
                                value_part = line.split('=')[1].strip()
                                coeffs_parsed.append(float(value_part))
                    
                    elif "Format 2" in coeff_format:
                        # Parse Format 2: 'b_coeff': [...] format
                        import re
                        match = re.search(r"'b_coeff':\s*\[(.*?)\]", coeff_input, re.DOTALL)
                        if match:
                            coeff_str = match.group(1)
                            # Extract coefficient values
                            coeff_values = re.findall(r"['\"]([+-]?[0-9]*\.?[0-9]+[eE]?[+-]?[0-9]*)['\"]", coeff_str)
                            coeffs_parsed = [float(val.replace('+', '')) for val in coeff_values]
                    
                    elif "Format 3" in coeff_format:
                        # Parse Format 3: [...] format
                        import re
                        match = re.search(r"\[(.*?)\]", coeff_input, re.DOTALL)
                        if match:
                            coeff_str = match.group(1)
                            # Extract coefficient values
                            coeff_values = re.findall(r"['\"]([+-]?[0-9]*\.?[0-9]+[eE]?[+-]?[0-9]*)['\"]", coeff_str)
                            coeffs_parsed = [float(val.replace('+', '')) for val in coeff_values]
                    
                    if coeffs_parsed and len(coeffs_parsed) > 0:
                        # Store parsed coefficients in session state
                        st.session_state.verified_coeffs = coeffs_parsed
                        st.session_state.show_verification_plots = True
                        st.success(f"✅ Successfully parsed {len(coeffs_parsed)} coefficients!")
                        
                        # Show parsed coefficients
                        st.subheader("Parsed Coefficients:")
                        for i, coeff in enumerate(coeffs_parsed):
                            st.text(f"B_{i} = {coeff:.6e}")
                        
                    else:
                        st.error("❌ Could not parse coefficients. Please check the format.")
                        
                except Exception as e:
                    st.error(f"❌ Error parsing coefficients: {str(e)}")
            else:
                st.warning("⚠️ Please enter coefficients to verify.")

# Main content area
if uploaded_file is not None:
    # Read data
    pixel_index, spectrum_data = processor.read_csv_data(uploaded_file)
    
    if pixel_index is not None and spectrum_data is not None:
        st.success(f"✅ Data loaded successfully! {len(pixel_index)} data points")
        
        # Initialize session state for peaks and ROI if not exists
        if 'matched_pixels' not in st.session_state:
            # Start with some default peaks
            st.session_state.matched_pixels = processor.default_pixel_indexs
            st.session_state.matched_wavenumbers = processor.default_ethanol_peaks
        
        # Initialize ROI for each peak
        if 'peak_rois' not in st.session_state:
            st.session_state.peak_rois = {}
            for i in range(len(st.session_state.matched_pixels)):
                peak_pixel = st.session_state.matched_pixels[i]
                roi_size = 100  # Default ROI size around each peak
                st.session_state.peak_rois[i] = {
                    'min': max(int(peak_pixel - roi_size), int(min(pixel_index))),
                    'max': min(int(peak_pixel + roi_size), int(max(pixel_index)))
                }
        
        # Main spectrum plot (full view)
        st.header("📊 Full Raman Spectrum with Peaks")
        
        fig_main = go.Figure()
        
        # Full spectrum
        fig_main.add_trace(
            go.Scatter(
                x=pixel_index,
                y=spectrum_data,
                mode='lines',
                name='Raman Spectrum',
                line=dict(color='lightblue', width=2)
            )
        )
        
        # Add peak markers and ROI regions
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        for i, (pixel, wavenumber) in enumerate(zip(st.session_state.matched_pixels, 
                                                    st.session_state.matched_wavenumbers)):
            color = colors[i % len(colors)]
            spectrum_intensity = np.interp(pixel, pixel_index, spectrum_data)
            
            # Peak marker
            fig_main.add_trace(
                go.Scatter(
                    x=[pixel],
                    y=[spectrum_intensity],
                    mode='markers+text',
                    name=f'Peak {i+1}: {wavenumber}cm⁻¹',
                    marker=dict(color=color, size=12, line=dict(width=2, color='white')),
                    text=[f'P{i+1}'],
                    textposition="top center",
                    textfont=dict(size=12, color='white')
                )
            )
            
            # Peak line
            fig_main.add_vline(
                x=pixel,
                line_dash="dash",
                line_color=color,
                line_width=2,
                opacity=0.8
            )
            
            # ROI region highlight
            if i in st.session_state.peak_rois:
                roi = st.session_state.peak_rois[i]
                roi_mask = (pixel_index >= roi['min']) & (pixel_index <= roi['max'])
                fig_main.add_trace(
                    go.Scatter(
                        x=pixel_index[roi_mask],
                        y=spectrum_data[roi_mask],
                        mode='lines',
                        name=f'ROI {i+1}',
                        line=dict(color=color, width=4),
                        opacity=0.6,
                        showlegend=False
                    )
                )
        
        fig_main.update_layout(
            title="Raman Spectrum with Peak Positions and ROI Regions",
            xaxis_title="Pixel Position",
            yaxis_title="Intensity",
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
        
        # Peak adjustment interface
        st.header("🎯 Peak Adjustment Controls")
        
        # # Add/Remove peak buttons
        col_control1, col_control2, col_control3 = st.columns(3)
        with col_control1:
            if st.button("➕ Add Peak", use_container_width=True):
                # Add peak at center of spectrum
                center_pixel = len(pixel_index) // 2
                st.session_state.matched_pixels.append(center_pixel)
                st.session_state.matched_wavenumbers.append(1000)
                # Add ROI for new peak
                new_peak_idx = len(st.session_state.matched_pixels) - 1
                roi_size = 100
                st.session_state.peak_rois[new_peak_idx] = {
                    'min': max(int(center_pixel - roi_size), int(min(pixel_index))),
                    'max': min(int(center_pixel + roi_size), int(max(pixel_index)))
                }
                st.rerun()
        
        with col_control2:
            if st.button("➖ Remove Last Peak", use_container_width=True):
                if len(st.session_state.matched_pixels) > 1:
                    last_idx = len(st.session_state.matched_pixels) - 1
                    st.session_state.matched_pixels.pop()
                    st.session_state.matched_wavenumbers.pop()
                    if last_idx in st.session_state.peak_rois:
                        del st.session_state.peak_rois[last_idx]
                    st.rerun()
        
        with col_control3:
            if st.button("🔄 Reset All Peaks", use_container_width=True):
                st.session_state.matched_pixels = [100, 300, 500, 700]
                st.session_state.matched_wavenumbers = [2973, 2927, 1455, 884]
                st.session_state.peak_rois = {}
                for i in range(len(st.session_state.matched_pixels)):
                    peak_pixel = st.session_state.matched_pixels[i]
                    roi_size = 100
                    st.session_state.peak_rois[i] = {
                        'min': max(int(peak_pixel - roi_size), int(min(pixel_index))),
                        'max': min(int(peak_pixel + roi_size), int(max(pixel_index)))
                    }
                st.rerun()
        
       
        
        # Individual peak adjustment with ROI plots
        st.subheader("🔍 Individual Peak Adjustment with ROI")
        
        # Create tabs for each peak
        if len(st.session_state.matched_pixels) > 0:
            tab_labels = [f"Peak {i+1}" for i in range(len(st.session_state.matched_pixels))]
            tabs = st.tabs(tab_labels)
            
            for i, tab in enumerate(tabs):
                with tab:
                    # Initialize ROI if not exists
                    if i not in st.session_state.peak_rois:
                        peak_pixel = st.session_state.matched_pixels[i]
                        roi_size = 100
                        st.session_state.peak_rois[i] = {
                            'min': max(int(peak_pixel - roi_size), int(min(pixel_index))),
                            'max': min(int(peak_pixel + roi_size), int(max(pixel_index)))
                        }
                    
                    # Top row: Peak Position and Wavenumber side by side
                    st.markdown(f"**🎯 Peak {i+1} Controls**")
                    col_peak_pos, col_wavenumber = st.columns(2)
                    
                    with col_peak_pos:
                        st.markdown("**Peak Position:**")
                        
                        # Get current pixel position
                        current_pixel = st.session_state.matched_pixels[i]
                        
                        # Peak position input (main control)
                        new_pixel = st.number_input(
                            "Pixel Position",
                            min_value=int(min(pixel_index)),
                            max_value=int(max(pixel_index)),
                            value=int(current_pixel),
                            step=1,
                            key=f"pixel_input_{i}",
                            help="Enter pixel position - ROI will auto-adjust to ±100 pixels"
                        )
                        
                        # Update pixel position and auto-adjust ROI
                        if new_pixel != current_pixel:
                            st.session_state.matched_pixels[i] = new_pixel
                            # Auto-adjust ROI to ±100 pixels around the new peak position
                            roi_size = 100
                            new_roi_min = max(int(new_pixel - roi_size), int(min(pixel_index)))
                            new_roi_max = min(int(new_pixel + roi_size), int(max(pixel_index)))
                            st.session_state.peak_rois[i]['min'] = new_roi_min
                            st.session_state.peak_rois[i]['max'] = new_roi_max
                        
                        # Fine adjustment buttons
                        fine_col1, fine_col2, fine_col3, fine_col4 = st.columns(4)
                        with fine_col1:
                            if st.button("⬅️10", key=f"fine_left_10_{i}"):
                                new_pos = max(st.session_state.matched_pixels[i] - 10, int(min(pixel_index)))
                                st.session_state.matched_pixels[i] = new_pos
                                # Auto-adjust ROI
                                roi_size = 100
                                st.session_state.peak_rois[i]['min'] = max(int(new_pos - roi_size), int(min(pixel_index)))
                                st.session_state.peak_rois[i]['max'] = min(int(new_pos + roi_size), int(max(pixel_index)))
                                st.rerun()
                        with fine_col2:
                            if st.button("◀️1", key=f"fine_left_1_{i}"):
                                new_pos = max(st.session_state.matched_pixels[i] - 1, int(min(pixel_index)))
                                st.session_state.matched_pixels[i] = new_pos
                                # Auto-adjust ROI
                                roi_size = 100
                                st.session_state.peak_rois[i]['min'] = max(int(new_pos - roi_size), int(min(pixel_index)))
                                st.session_state.peak_rois[i]['max'] = min(int(new_pos + roi_size), int(max(pixel_index)))
                                st.rerun()
                        with fine_col3:
                            if st.button("▶️1", key=f"fine_right_1_{i}"):
                                new_pos = min(st.session_state.matched_pixels[i] + 1, int(max(pixel_index)))
                                st.session_state.matched_pixels[i] = new_pos
                                # Auto-adjust ROI
                                roi_size = 100
                                st.session_state.peak_rois[i]['min'] = max(int(new_pos - roi_size), int(min(pixel_index)))
                                st.session_state.peak_rois[i]['max'] = min(int(new_pos + roi_size), int(max(pixel_index)))
                                st.rerun()
                        with fine_col4:
                            if st.button("➡️10", key=f"fine_right_10_{i}"):
                                new_pos = min(st.session_state.matched_pixels[i] + 10, int(max(pixel_index)))
                                st.session_state.matched_pixels[i] = new_pos
                                # Auto-adjust ROI
                                roi_size = 100
                                st.session_state.peak_rois[i]['min'] = max(int(new_pos - roi_size), int(min(pixel_index)))
                                st.session_state.peak_rois[i]['max'] = min(int(new_pos + roi_size), int(max(pixel_index)))
                                st.rerun()
                    
                    with col_wavenumber:
                        st.markdown("**Wavenumber:**")
                        
                        # Wavenumber input
                        new_wavenumber = st.number_input(
                            "Wavenumber (cm⁻¹)",
                            value=float(st.session_state.matched_wavenumbers[i]),
                            step=0.1,
                            format="%.1f",
                            key=f"wavenumber_input_{i}"
                        )
                        st.session_state.matched_wavenumbers[i] = new_wavenumber
                        
                        # Show current wavelength
                        current_wavelength = processor.wavenumber_to_wavelength(new_wavenumber)
                        st.caption(f"Wavelength: {current_wavelength:.2f} nm")
                        
                        # Peak info
                        spectrum_intensity = np.interp(st.session_state.matched_pixels[i], pixel_index, spectrum_data)
                        st.caption(f"Peak Intensity: {spectrum_intensity:.1f}")
                        # st.info(f"**Peak {i+1} Info:**\n"
                        #         f"- Pixel: {st.session_state.matched_pixels[i]}\n"
                        #         f"- Intensity: {spectrum_intensity:.1f}\n"
                        #         f"- Wavenumber: {new_wavenumber} cm⁻¹\n"
                        #         f"- Wavelength: {current_wavelength:.2f} nm")
                    # st.newline()
                    # st.divider()
                    
                    # Bottom row: ROI Settings (left) and ROI View (right)
                    col_roi_settings, col_roi_plot = st.columns([1, 3])
                    
                    with col_roi_settings:
                        st.markdown("**ROI Settings:**")
                        
                        roi = st.session_state.peak_rois[i]
                        
                        # ROI range inputs
                        roi_min = st.number_input(
                            "ROI Min",
                            min_value=int(min(pixel_index)),
                            max_value=int(max(pixel_index)),
                            value=roi['min'],
                            step=1,
                            key=f"roi_min_{i}"
                        )
                        
                        roi_max = st.number_input(
                            "ROI Max",
                            min_value=int(min(pixel_index)),
                            max_value=int(max(pixel_index)),
                            value=roi['max'],
                            step=1,
                            key=f"roi_max_{i}"
                        )
                        
                        # Ensure ROI min < ROI max
                        if roi_min >= roi_max:
                            st.error("⚠️ ROI Min must be less than ROI Max")
                        else:
                            st.session_state.peak_rois[i]['min'] = roi_min
                            st.session_state.peak_rois[i]['max'] = roi_max
                        
                        # Quick ROI adjustment buttons
                        roi_btn_col1, roi_btn_col2 = st.columns(2)
                        with roi_btn_col1:
                            if st.button("🔍 Zoom In", key=f"zoom_in_{i}"):
                                current_range = roi_max - roi_min
                                center = (roi_min + roi_max) / 2
                                new_range = max(current_range * 0.7, 20)
                                new_min = max(int(center - new_range/2), int(min(pixel_index)))
                                new_max = min(int(center + new_range/2), int(max(pixel_index)))
                                st.session_state.peak_rois[i]['min'] = new_min
                                st.session_state.peak_rois[i]['max'] = new_max
                                st.rerun()
                        
                        with roi_btn_col2:
                            if st.button("🔍 Zoom Out", key=f"zoom_out_{i}"):
                                current_range = roi_max - roi_min
                                center = (roi_min + roi_max) / 2
                                new_range = min(current_range * 1.5, len(pixel_index))
                                new_min = max(int(center - new_range/2), int(min(pixel_index)))
                                new_max = min(int(center + new_range/2), int(max(pixel_index)))
                                st.session_state.peak_rois[i]['min'] = new_min
                                st.session_state.peak_rois[i]['max'] = new_max
                                st.rerun()
                        
                        if st.button("🎯 Auto ROI", key=f"auto_roi_{i}"):
                            # Auto-adjust ROI to ±100 pixels around current peak
                            peak_pixel = st.session_state.matched_pixels[i]
                            roi_size = 100
                            new_min = max(int(peak_pixel - roi_size), int(min(pixel_index)))
                            new_max = min(int(peak_pixel + roi_size), int(max(pixel_index)))
                            st.session_state.peak_rois[i]['min'] = new_min
                            st.session_state.peak_rois[i]['max'] = new_max
                            st.rerun()
                        
                        st.caption(f"ROI Range: {roi_max - roi_min} pixels")
                    
                    with col_roi_plot:
                        # ROI plot for this peak
                        roi = st.session_state.peak_rois[i]
                        roi_mask = (pixel_index >= roi['min']) & (pixel_index <= roi['max'])
                        
                        fig_roi = go.Figure()
                        
                        # ROI spectrum
                        fig_roi.add_trace(
                            go.Scatter(
                                x=pixel_index[roi_mask],
                                y=spectrum_data[roi_mask],
                                mode='lines',
                                name=f'ROI Spectrum',
                                line=dict(color='blue', width=2)
                            )
                        )
                        
                        # Peak marker
                        color = colors[i % len(colors)]
                        peak_pixel = st.session_state.matched_pixels[i]
                        spectrum_intensity = np.interp(peak_pixel, pixel_index, spectrum_data)
                        
                        fig_roi.add_trace(
                            go.Scatter(
                                x=[peak_pixel],
                                y=[spectrum_intensity],
                                mode='markers+text',
                                name=f'Peak {i+1}',
                                marker=dict(color=color, size=15, line=dict(width=2, color='white')),
                                text=[f'P{i+1}'],
                                textposition="top center",
                                textfont=dict(size=14, color='white')
                            )
                        )
                        
                        # Peak line
                        fig_roi.add_vline(
                            x=peak_pixel,
                            line_dash="dash",
                            line_color=color,
                            line_width=3
                        )
                        
                        fig_roi.update_layout(
                            title=f"Peak {i+1} ROI View (Pixels {roi['min']}-{roi['max']})",
                            xaxis_title="Pixel Position",
                            yaxis_title="Intensity",
                            height=400,
                            xaxis=dict(range=[roi['min'], roi['max']])
                        )
                        
                        st.plotly_chart(fig_roi, use_container_width=True)
        
        st.divider()
        # Detailed results
        st.subheader("Peak Matching Results")
        peak_df = pd.DataFrame({
            'Peak': [f"Peak {i+1}" for i in range(len(st.session_state.matched_pixels))],
            'Pixel Position': [f"{p:.1f}" for p in st.session_state.matched_pixels],
            'Wavelength (nm)': [f"{processor.wavenumber_to_wavelength(wn):.2f}" 
                                for wn in st.session_state.matched_wavenumbers],
            'Wavenumber (cm⁻¹)': st.session_state.matched_wavenumbers,
        })
        st.dataframe(peak_df, use_container_width=True)

        st.divider()
        # Results section
        if len(st.session_state.matched_pixels) >= 2:
            st.header("📊 Calibration Results")
            
            matched_wavelengths = [processor.wavenumber_to_wavelength(wn) 
                                    for wn in st.session_state.matched_wavenumbers]
            
            coeffs, degree, fitting_results = processor.polynomial_fitting(
                st.session_state.matched_pixels, matched_wavelengths
            )
            
            if coeffs is not None:
                # Results summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Laser Wavelength", f"{laser_wavelength} nm")
                with col2:
                    st.metric("Number of Peaks", len(st.session_state.matched_pixels))
                with col3:
                    st.metric("Polynomial Degree", degree)

                
                coeff_col1, coeff_col2 = st.columns(2)
                with coeff_col1:
                    st.subheader("Polynomial Fitting")
                    # Show fitting plot
                    fig_fit = go.Figure()
                    
                    # Scatter plot of peak positions
                    fig_fit.add_trace(
                        go.Scatter(
                            x=st.session_state.matched_pixels,
                            y=matched_wavelengths,
                            mode='markers',
                            name='Peak Positions',
                            marker=dict(color='red', size=10)
                        )
                    )
                    
                    # Fitting line
                    x_fit = np.linspace(min(st.session_state.matched_pixels), 
                                        max(st.session_state.matched_pixels), 1000)
                    y_fit = np.polyval(coeffs, x_fit)
                    
                    fig_fit.add_trace(
                        go.Scatter(
                            x=x_fit,
                            y=y_fit,
                            mode='lines',
                            name=f'{degree}-degree fit',
                            line=dict(color='blue', width=2)
                        )
                    )
                    
                    fig_fit.update_layout(
                        title=f"Polynomial Fitting (Degree {degree})",
                        xaxis_title="Pixel Position",
                        yaxis_title="Wavelength (nm)",
                        height=400,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_fit, use_container_width=True)
                
                with coeff_col2:
                    st.subheader("Polynomial Coefficients")
                    # Create coefficients dataframe
                    coeff_data = []
                    for i, coeff in enumerate(coeffs[::-1]):
                        coeff_data.append({
                            'Coefficient': f'B_{i}',
                            'Value': f'{coeff:.6e}',
                            'Description': f'x^{i}' if i > 0 else 'constant'
                        })
                    
                    coeffs_df = pd.DataFrame(coeff_data)
                    st.write(coeffs_df)
                    
                    # Also show the polynomial equation
                    st.write("**Polynomial Equation:**")
                    st.write('y represents the wavelength in nm as a function of pixel position x')

                    equation_parts = []
                    for i, coeff in enumerate(coeffs[::-1]):
                        if i == 0:
                            equation_parts.append(f"{coeff:.3e}")
                        elif i == 1:
                            equation_parts.append(f"{coeff:.3e}x")
                        else:
                            equation_parts.append(f"{coeff:.3e}x^{i}")
                    
                    equation = " + ".join(equation_parts)
                    st.code(f"y = {equation}", language='python')

                # Plot pixel vs wavenumber
                st.subheader("Pixel vs Wavenumber Using Polynomial Fit")
                st.write("This plot shows the relationship between pixel positions and wavenumbers based on the polynomial fit.")
                # x should be 0 - pixel_index[-1]
                x_values = processor.pixel_indexs
                y_values = np.polyval(coeffs, x_values)
                fig_pixel_vs_wn = go.Figure()
                fig_pixel_vs_wn.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='lines',
                        name='Pixel vs Wavenumber',
                        line=dict(color='purple', width=2)
                    )
                )
                fig_pixel_vs_wn.update_layout(
                    title="Pixel Position vs Wavenumber",
                    xaxis_title="Pixel Position",
                    yaxis_title="Wavenumber (cm⁻¹)",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_pixel_vs_wn, use_container_width=True)
              
                
                
           
                
                # Export section
                st.subheader("🚀 Export Results")

                # Create a json like format for demo_server.py
                # Laser wavelength and polynomial coefficients
               
                # Format for demo_server.py
                formatted_coeffs = [f'"{coeff:.7E}"' for coeff in coeffs[::-1]]
                server_format = f"'b_coeff': [{', '.join(formatted_coeffs)}]"
                server_format += f",\n'laser_wavelength': {laser_wavelength},\n'degree': {degree}"
                
                st.code(server_format, language='python')
                
                # Download button
                results_text = f"""Laser wavelength: {laser_wavelength} nm
                                Polynomial degree: {degree}
                                Number of peaks: {len(st.session_state.matched_pixels)}

                                Peak matching results:
                                """
                for i, (pixel, wavenumber) in enumerate(zip(st.session_state.matched_pixels, 
                                                            st.session_state.matched_wavenumbers)):
                    wavelength = processor.wavenumber_to_wavelength(wavenumber)
                    results_text += f"Peak {i+1}: pixel={pixel:.1f}, wavenumber={wavenumber}, wavelength={wavelength:.2f}\n"
                
                results_text += f"\nPolynomial coefficients:\n"
                for i, coeff in enumerate(coeffs[::-1]):
                    results_text += f"B_{i} = {coeff:.10e}\n"
                
                
                st.download_button(
                    label="📥 Download Calibration Results",
                    data=results_text,
                    file_name=f"calibration_results_{laser_wavelength}nm.txt",
                    mime="text/plain"
                )

    # Add verification plots section after the main content
    if 'show_verification_plots' in st.session_state and st.session_state.show_verification_plots:
        st.divider()
        st.header("🔍 Coefficient Verification Plots")
        st.markdown("These plots show the pixel-wavelength and pixel-wavenumber relationships using the loaded coefficients.")
        
        if 'verified_coeffs' in st.session_state:
            coeffs = st.session_state.verified_coeffs
            
            # Generate pixel range (use full range if data is available, otherwise use default)
            if pixel_index is not None:
                x_pixels = np.linspace(min(pixel_index), max(pixel_index), 1000)
            else:
                x_pixels = np.linspace(0, 2048, 1000)  # Default range
            
            # Calculate wavelengths using the polynomial
            # Note: coefficients are in ascending order (B_0, B_1, B_2, ...)
            # but numpy.polyval expects descending order, so we reverse
            wavelengths = np.polyval(coeffs[::-1], x_pixels)
            
            # Calculate wavenumbers from wavelengths
            wavenumbers = processor.wavelength_to_wavenumber(wavelengths)
            
            # Create two columns for the plots
            plot_col1, plot_col2 = st.columns(2)
            
            with plot_col1:
                # Pixel vs Wavelength plot
                fig_wave = go.Figure()
                fig_wave.add_trace(
                    go.Scatter(
                        x=x_pixels,
                        y=wavelengths,
                        mode='lines',
                        name='Pixel vs Wavelength',
                        line=dict(color='blue', width=2)
                    )
                )
                
                # Add markers for known peaks if they exist
                if 'matched_pixels' in st.session_state and 'matched_wavenumbers' in st.session_state:
                    peak_wavelengths = [processor.wavenumber_to_wavelength(wn) 
                                       for wn in st.session_state.matched_wavenumbers]
                    fig_wave.add_trace(
                        go.Scatter(
                            x=st.session_state.matched_pixels,
                            y=peak_wavelengths,
                            mode='markers',
                            name='Known Peaks',
                            marker=dict(color='red', size=8)
                        )
                    )
                
                fig_wave.update_layout(
                    title="Pixel Position vs Wavelength",
                    xaxis_title="Pixel Position",
                    yaxis_title="Wavelength (nm)",
                    height=400
                )
                st.plotly_chart(fig_wave, use_container_width=True)
            
            with plot_col2:
                # Pixel vs Wavenumber plot
                fig_wn = go.Figure()
                fig_wn.add_trace(
                    go.Scatter(
                        x=x_pixels,
                        y=wavenumbers,
                        mode='lines',
                        name='Pixel vs Wavenumber',
                        line=dict(color='green', width=2)
                    )
                )
                
                # Add markers for known peaks if they exist
                if 'matched_pixels' in st.session_state and 'matched_wavenumbers' in st.session_state:
                    fig_wn.add_trace(
                        go.Scatter(
                            x=st.session_state.matched_pixels,
                            y=st.session_state.matched_wavenumbers,
                            mode='markers',
                            name='Known Peaks',
                            marker=dict(color='red', size=8)
                        )
                    )
                
                fig_wn.update_layout(
                    title="Pixel Position vs Wavenumber",
                    xaxis_title="Pixel Position",
                    yaxis_title="Wavenumber (cm⁻¹)",
                    height=400
                )
                st.plotly_chart(fig_wn, use_container_width=True)
            
            # Show coefficient information
            st.subheader("📊 Coefficient Information")
            coeff_info_col1, coeff_info_col2 = st.columns(2)
            
            with coeff_info_col1:
                st.metric("Polynomial Degree", len(coeffs) - 1)
                st.metric("Number of Coefficients", len(coeffs))
            
            with coeff_info_col2:
                if pixel_index is not None:
                    st.metric("Wavelength Range", f"{wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
                    st.metric("Wavenumber Range", f"{wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
            
            # Detailed coefficient table
            st.subheader("📋 Detailed Coefficients")
            coeff_df = pd.DataFrame({
                'Coefficient': [f'B_{i}' for i in range(len(coeffs))],
                'Value': [f'{coeff:.10e}' for coeff in coeffs],
                'Description': [f'x^{i}' if i > 0 else 'constant' for i in range(len(coeffs))]
            })
            st.dataframe(coeff_df, use_container_width=True)
            
            # Clear verification plots button
            if st.button("🗑️ Clear Verification Plots"):
                st.session_state.show_verification_plots = False
                if 'verified_coeffs' in st.session_state:
                    del st.session_state.verified_coeffs
                st.rerun()

else:
    st.info("👆 Please upload a CSV file to begin calibration")
    
    # Show coefficient verification plots when no file is uploaded
    if 'show_verification_plots' in st.session_state and st.session_state.show_verification_plots:
        st.divider()
        st.header("🔍 Coefficient Verification Plots")
        st.markdown("These plots show the pixel-wavelength and pixel-wavenumber relationships using the loaded coefficients.")
        
        if 'verified_coeffs' in st.session_state:
            coeffs = st.session_state.verified_coeffs
            
            # Generate pixel range (use default range when no data is available)
            x_pixels = np.linspace(0, 2048, 1000)  # Default range
            
            # Calculate wavelengths using the polynomial
            # Note: coefficients are in ascending order (B_0, B_1, B_2, ...)
            # but numpy.polyval expects descending order, so we reverse
            wavelengths = np.polyval(coeffs[::-1], x_pixels)
            
            # Calculate wavenumbers from wavelengths
            wavenumbers = processor.wavelength_to_wavenumber(wavelengths)
            
            # Create two columns for the plots
            plot_col1, plot_col2 = st.columns(2)
            
            with plot_col1:
                # Pixel vs Wavelength plot
                fig_wave = go.Figure()
                fig_wave.add_trace(
                    go.Scatter(
                        x=x_pixels,
                        y=wavelengths,
                        mode='lines',
                        name='Pixel vs Wavelength',
                        line=dict(color='blue', width=2)
                    )
                )
                
                fig_wave.update_layout(
                    title="Pixel Position vs Wavelength",
                    xaxis_title="Pixel Position",
                    yaxis_title="Wavelength (nm)",
                    height=400
                )
                st.plotly_chart(fig_wave, use_container_width=True)
            
            with plot_col2:
                # Pixel vs Wavenumber plot
                fig_wn = go.Figure()
                fig_wn.add_trace(
                    go.Scatter(
                        x=x_pixels,
                        y=wavenumbers,
                        mode='lines',
                        name='Pixel vs Wavenumber',
                        line=dict(color='green', width=2)
                    )
                )
                
                fig_wn.update_layout(
                    title="Pixel Position vs Wavenumber",
                    xaxis_title="Pixel Position",
                    yaxis_title="Wavenumber (cm⁻¹)",
                    height=400
                )
                st.plotly_chart(fig_wn, use_container_width=True)
            
            # Show coefficient information
            st.subheader("📊 Coefficient Information")
            coeff_info_col1, coeff_info_col2 = st.columns(2)
            
            with coeff_info_col1:
                st.metric("Polynomial Degree", len(coeffs) - 1)
                st.metric("Number of Coefficients", len(coeffs))
            
            with coeff_info_col2:
                st.metric("Wavelength Range", f"{wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
                st.metric("Wavenumber Range", f"{wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹")
            
            # Detailed coefficient table
            st.subheader("📋 Detailed Coefficients")
            coeff_df = pd.DataFrame({
                'Coefficient': [f'B_{i}' for i in range(len(coeffs))],
                'Value': [f'{coeff:.10e}' for coeff in coeffs],
                'Description': [f'x^{i}' if i > 0 else 'constant' for i in range(len(coeffs))]
            })
            st.dataframe(coeff_df, use_container_width=True)
            
            # Clear verification plots button
            if st.button("🗑️ Clear Verification Plots"):
                st.session_state.show_verification_plots = False
                if 'verified_coeffs' in st.session_state:
                    del st.session_state.verified_coeffs
                st.rerun()
    
    # Show example of expected CSV format
    st.subheader("📋 Expected CSV Format")
    example_df = pd.DataFrame({
        'PixelIndex': [0, 1, 2, 3, 4],
        'WaveNumber': [500, 501, 502, 503, 504],
        'DarkSpectrum': [100, 101, 102, 103, 104],
        'Spectrum1': [1000, 1100, 1200, 1300, 1400],
        'Spectrum2': [1050, 1150, 1250, 1350, 1450],
        'Spectrum3': [980, 1080, 1180, 1280, 1380]
    })
    st.dataframe(example_df)
    st.caption("The CSV should contain PixelIndex, WaveNumber, DarkSpectrum columns, followed by spectrum data columns")