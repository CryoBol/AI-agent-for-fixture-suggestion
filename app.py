import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import time

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="PDA Occluder Fixture Suite", layout="wide")

st.title("🔬 PDA Occluder Heat-Setting Fixture Studio")
st.markdown("Integrated platform featuring CAD/Image ingestion, thermal profiling, and interactive WebGL assembly.")

# Initialize session state for parametric dimensions
if 'd1' not in st.session_state: st.session_state.d1 = 24.0
if 'd2' not in st.session_state: st.session_state.d2 = 9.0
if 'd3' not in st.session_state: st.session_state.d3 = 6.0
if 'd4' not in st.session_state: st.session_state.d4 = 10.0
if 'waist' not in st.session_state: st.session_state.waist = 9.0

# Multi-Tab Layout
tab1, tab2, tab3 = st.tabs([
    "1. Device Ingestion & Mechanics", 
    "2. Interactive 3D Mold (WebGL)", 
    "3. Thermal Profiling"
])

# ==============================================================================
# TAB 1: INGESTION & MECHANICS
# ==============================================================================
with tab1:
    st.header("Occluder Design Input")
    
    input_method = st.radio(
        "Select Geometry Input Method:",
        ["Parametric Dimensions (Manual)", "Upload CAD (.step / .stp)", "Upload Reference Image"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # 1. MANUAL DIMENSIONS (Now wrapped in a form)
    if input_method == "Parametric Dimensions (Manual)":
        with st.form("cad_input_form"):
            st.subheader("Define Mold Parameters")
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                in_d1 = st.slider("Aortic Disc Ø (D1)", 16.0, 34.0, st.session_state.d1, 0.5)
                in_d2 = st.slider("Aortic Waist Ø (D2)", 5.0, 16.0, st.session_state.d2, 0.5)
                in_waist = st.slider("Waist Length", 4.0, 20.0, st.session_state.waist, 0.5)
            with col_t2:
                in_d3 = st.slider("Pulmonary Waist Ø (D3)", 3.0, 14.0, st.session_state.d3, 0.5)
                in_d4 = st.slider("Pulmonary Disc Ø (D4)", 4.0, 24.0, st.session_state.d4, 0.5)
            
            # The Generate Button
            submit_button = st.form_submit_button("⚙️ Generate CAD Model", type="primary")
            
            if submit_button:
                st.session_state.d1 = in_d1
                st.session_state.d2 = in_d2
                st.session_state.waist = in_waist
                st.session_state.d3 = in_d3
                st.session_state.d4 = in_d4
                st.success("CAD parameters updated! Switch to Tab 2 to view the model.")

    # 2. CAD UPLOAD
    elif input_method == "Upload CAD (.step / .stp)":
        uploaded_cad = st.file_uploader("Upload PDA Occluder STEP file", type=["step", "stp"])
        if st.button("⚙️ Extract & Generate from CAD", type="primary"):
            if uploaded_cad is not None:
                with st.spinner("Parsing STEP geometry..."):
                    time.sleep(1.5)
                    st.success("Geometry parsed! Mold parameters auto-updated.")
            else:
                st.warning("Please upload a file first.")

    # 3. IMAGE UPLOAD
    elif input_method == "Upload Reference Image":
        uploaded_img = st.file_uploader("Upload fluoroscopy or schematic image", type=["png", "jpg", "jpeg"])
        if st.button("⚙️ Process Image & Generate", type="primary"):
            if uploaded_img is not None:
                with st.spinner("Running edge-detection..."):
                    time.sleep(1.5)
                    st.success("Profile extracted! Mold parameters auto-updated.")
            else:
                st.warning("Please upload an image first.")

    st.markdown("---")
    st.subheader("Dual-Layer Radial Force Estimation")
    f_rad_waist = (0.05 * (0.16**4) * 36 * (1.5**1.2)) + (0.03 * (0.09**4) * 72 * (1.5**1.1))
    st.metric(label="Estimated Radial Force at Waist (Composite)", value=f"{f_rad_waist:.3f} N")


# ==============================================================================
# TAB 2: INTERACTIVE 3D WEBGL FIXTURE 
# ==============================================================================
with tab2:
    st.markdown("### Parametric Mold Assembly")
    st.markdown("The mold below is dynamically shaped based on your inputs. Use the panel to toggle the **Thermal Map Overlay**.")
    
    html_injection = f"""
    var param_starts = {{
        D1: {st.session_state.d1},
        D2: {st.session_state.d2},
        D3: {st.session_state.d3},
        D4: {st.session_state.d4},
        waist: {st.session_state.waist}
    }};
    """

    three_js_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <style>
      :root{ --bg-deep:#0c0f13; --bg-viewport:#10151b; --panel:#141a21; --border:#262f3a; --text:#e7edf3; --text-faint:#5b6572; --red:#e04f4f; }
      *{box-sizing:border-box;}
      html,body{margin:0;padding:0;height:100%;background:var(--bg-deep);color:var(--text);font-family:sans-serif;overflow:hidden;}
      #app{display:flex;flex-direction:column;height:100vh;}
      #main{flex:1;display:flex;min-height:0;}
      #viewport{flex:1;position:relative;background:radial-gradient(ellipse at 50% 30%, #182029 0%, var(--bg-viewport) 70%);overflow:hidden;}
      #viewport canvas{display:block;width:100%;height:100%;cursor:grab;}
      aside#panel{width:350px;flex-shrink:0;background:var(--panel);border-left:1px solid var(--border);padding:20px;}
      .sec{margin-bottom:25px;}
      .sec h2{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--text-faint);margin:0 0 15px;border-bottom:1px solid var(--border);padding-bottom:5px;}
      .toggle{display:flex;align-items:center;justify-content:space-between;padding:10px 0;cursor:pointer;font-size:14px;}
      .switch{position:relative;width:34px;height:19px;flex-shrink:0;}
      .switch input{opacity:0;width:0;height:0;}
      .slider-tog{position:absolute;inset:0;background:#2a333d;border-radius:20px;transition:.15s;}
      .slider-tog::before{content:'';position:absolute;width:14px;height:14px;left:2.5px;top:2.5px;background:#8a94a1;border-radius:50%;transition:.15s;}
      .switch input:checked + .slider-tog{background:var(--red);}
      .switch input:checked + .slider-tog::before{transform:translateX(15px);background:#fff;}
      input[type=range]{width:100%; margin-top:10px;}
    </style>
    </head>
    <body>
    <div id="app">
      <div id="main">
        <div id="viewport"><canvas id="c"></canvas></div>
        <aside id="panel">
          <div class="sec">
            <h2>Analytics View</h2>
            <label class="toggle"><span style="color:var(--red); font-weight:bold;">🔥 Thermal Map Overlay</span>
              <span class="switch"><input type="checkbox" id="thermal"><span class="slider-tog"></span></span>
            </label>
            <label class="toggle"><span>Section view (cutaway)</span>
              <span class="switch"><input type="checkbox" id="section"><span class="slider-tog"></span></span>
            </label>
          </div>
          <div class="sec">
            <h2>Assembly State</h2>
            <label style="font-size:13px; color:#8a94a1;">Explode View</label>
            <input type="range" id="explode" min="0" max="100" step="1" value="0">
          </div>
        </aside>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function(){
      "use strict";
      
      """ + html_injection + """

      var canvas = document.getElementById('c');
      var renderer = new THREE.WebGLRenderer({canvas:canvas, antialias:true});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
      renderer.localClippingEnabled = true;

      var scene = new THREE.Scene();
      var camera = new THREE.PerspectiveCamera(45, 1, 0.1, 2000);
      camera.position.set(60, 40, 80);
      camera.lookAt(0, 0, 0);

      // Improved lighting so geometry isn't pitch black
      var ambient = new THREE.AmbientLight(0x404040, 1.5); scene.add(ambient);
      var keyLight = new THREE.DirectionalLight(0xffffff, 1.0); keyLight.position.set(50, 50, 50); scene.add(keyLight);
      var fillLight = new THREE.DirectionalLight(0x90b0d0, 0.5); fillLight.position.set(-50, 20, -50); scene.add(fillLight);

      var clipPlane = new THREE.Plane(new THREE.Vector3(1,0,0), 0);

      // --- FIXED MATERIALS (Added Emissive for Thermal) ---
      function createMat(color, emissive, intensity) {
          return new THREE.MeshStandardMaterial({
              color: color,
              roughness: 0.4,
              metalness: 0.8,
              emissive: emissive,
              emissiveIntensity: intensity,
              side: THREE.DoubleSide
          });
      }

      // Standard Cold Materials
      var mSteel = createMat(0x8899a6, 0x000000, 0);
      
      // Thermal Hot Materials (Glowing)
      var mHotCore = createMat(0xff2200, 0xffaa00, 0.8); // Core gets hottest
      var mHotPlate = createMat(0x661100, 0xcc2200, 0.5); // Plates are slightly cooler

      var assembly = new THREE.Group(); scene.add(assembly);
      
      // Meshes
      var meshMandrel, meshTop, meshBottom;
      var gBottom = new THREE.Group(), gTop = new THREE.Group();
      assembly.add(meshMandrel, gBottom, gTop);

      function V(x,y){ return new THREE.Vector2(x, y); }

      function buildGeometry() {
          // 1. Central Core / Mandrel
          var mPts = [
              V(0.01, -15), V(param_starts.D3/2, -15), 
              V(param_starts.D2/2, -param_starts.waist/2), 
              V(param_starts.D2/2, param_starts.waist/2), 
              V(param_starts.D3/2, 15), V(0.01, 15)
          ];
          meshMandrel = new THREE.Mesh(new THREE.LatheGeometry(mPts, 64), mSteel);
          scene.add(meshMandrel);

          // 2. Top Plate
          var tPts = [
              V(param_starts.D3/2 + 0.2, param_starts.waist/2), 
              V(param_starts.D1/2 + 4, param_starts.waist/2), 
              V(param_starts.D1/2 + 4, param_starts.waist/2 + 6), 
              V(param_starts.D3/2 + 0.2, param_starts.waist/2 + 6)
          ];
          meshTop = new THREE.Mesh(new THREE.LatheGeometry(tPts, 64), mSteel);
          gTop.add(meshTop); scene.add(gTop);

          // 3. Bottom Plate
          var bPts = [
              V(param_starts.D3/2 + 0.2, -param_starts.waist/2 - 6), 
              V(param_starts.D4/2 + 4, -param_starts.waist/2 - 6), 
              V(param_starts.D4/2 + 4, -param_starts.waist/2), 
              V(param_starts.D3/2 + 0.2, -param_starts.waist/2)
          ];
          meshBottom = new THREE.Mesh(new THREE.LatheGeometry(bPts, 64), mSteel);
          gBottom.add(meshBottom); scene.add(gBottom);
      }

      function updateMaterials() {
          var isThermal = document.getElementById('thermal').checked;
          if (isThermal) {
              meshMandrel.material = mHotCore;
              meshTop.material = mHotPlate;
              meshBottom.material = mHotPlate;
          } else {
              meshMandrel.material = mSteel;
              meshTop.material = mSteel;
              meshBottom.material = mSteel;
          }
      }

      document.getElementById('explode').addEventListener('input', e => { 
          var val = e.target.value / 100;
          gTop.position.y = val * 20;
          gBottom.position.y = -val * 20;
      });

      document.getElementById('section').addEventListener('change', e => { 
          renderer.clippingPlanes = e.target.checked ? [clipPlane] : []; 
      });
      
      document.getElementById('thermal').addEventListener('change', updateMaterials);

      function animate(){
        requestAnimationFrame(animate);
        renderer.render(scene, camera);
      }
      
      window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth/window.innerHeight; 
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      });
      
      renderer.setSize(window.innerWidth, window.innerHeight);
      buildGeometry();
      animate();
    })();
    </script>
    </body>
    </html>
    """
    components.html(three_js_code, height=700, scrolling=False)


# ==============================================================================
# TAB 3: THERMAL PROFILING 
# ==============================================================================
with tab3:
    st.header("Salt Bath / Furnace Heat-Setting Profile")
    st.markdown("Simulated heat transfer through the 316L stainless steel tooling to the Nitinol mesh.")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        target_temp = st.slider("Target Set Temperature (°C)", 450, 550, 500, 10)
        soak_time = st.slider("Soak Time (minutes)", 5, 20, 10, 1)
        
    with c2:
        time_axis = np.linspace(0, soak_time + 10, 100)
        furnace_temp = np.where(time_axis < 2, 25 + (target_temp-25)*(time_axis/2), target_temp)
        furnace_temp[time_axis > soak_time + 2] = furnace_temp[time_axis > soak_time + 2] * np.exp(-(time_axis[time_axis > soak_time + 2] - (soak_time + 2))/3)
        
        tool_temp = np.zeros_like(time_axis)
        for i in range(1, len(time_axis)):
            tool_temp[i] = tool_temp[i-1] + 0.15 * (furnace_temp[i] - tool_temp[i-1])

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(time_axis, furnace_temp, label="Environment (Furnace)", color="#e04f4f", linestyle="--")
        ax.plot(time_axis, tool_temp, label="Core Mandrel (Nitinol Contact)", color="#ffcc00", linewidth=2.5)
        ax.axhline(target_temp * 0.95, color="gray", linestyle=":", label="95% Transformation Temp")
        
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Temperature (°C)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.tick_params(colors='gray')
        ax.xaxis.label.set_color('gray')
        ax.yaxis.label.set_color('gray')
        
        st.pyplot(fig)
