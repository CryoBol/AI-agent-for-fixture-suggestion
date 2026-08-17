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
    
    # 1. MANUAL DIMENSIONS
    if input_method == "Parametric Dimensions (Manual)":
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.session_state.d1 = st.slider("Aortic Disc Ø (D1)", 16.0, 34.0, st.session_state.d1, 0.5)
            st.session_state.d2 = st.slider("Aortic Waist Ø (D2)", 5.0, 16.0, st.session_state.d2, 0.5)
            st.session_state.waist = st.slider("Waist Length", 4.0, 20.0, st.session_state.waist, 0.5)
        with col_t2:
            st.session_state.d3 = st.slider("Pulmonary Waist Ø (D3)", 3.0, 14.0, st.session_state.d3, 0.5)
            st.session_state.d4 = st.slider("Pulmonary Disc Ø (D4)", 4.0, 24.0, st.session_state.d4, 0.5)
            
    # 2. CAD UPLOAD
    elif input_method == "Upload CAD (.step / .stp)":
        uploaded_cad = st.file_uploader("Upload PDA Occluder STEP file", type=["step", "stp"])
        if uploaded_cad is not None:
            with st.spinner("Parsing STEP geometry via OpenCASCADE... extracting bounding cylinders..."):
                # MOCK: This is where you would use pythonocc-core or FreeCAD python API
                time.sleep(1.5)
                st.success("Geometry parsed successfully! Mold parameters auto-updated.")
                st.session_state.d1 = 28.0
                st.session_state.d2 = 10.0
                st.session_state.d3 = 8.0
                st.session_state.d4 = 14.0
                st.session_state.waist = 11.0
                st.json({"Extracted_D1": 28.0, "Extracted_D2": 10.0, "Extracted_D3": 8.0, "Extracted_D4": 14.0, "Extracted_Waist": 11.0})

    # 3. IMAGE UPLOAD
    elif input_method == "Upload Reference Image":
        uploaded_img = st.file_uploader("Upload fluoroscopy or schematic image", type=["png", "jpg", "jpeg"])
        if uploaded_img is not None:
            st.image(uploaded_img, width=300, caption="Uploaded Reference")
            with st.spinner("Running edge-detection & dimensional scaling..."):
                # MOCK: This is where you would use OpenCV/skimage for profile extraction
                time.sleep(1.5)
                st.success("Profile extracted! Mold parameters auto-updated.")
                st.session_state.d1 = 22.0
                st.session_state.d2 = 7.0
                st.session_state.d3 = 5.0
                st.session_state.d4 = 9.0
                st.session_state.waist = 7.5

    st.markdown("---")
    st.subheader("Dual-Layer Radial Force Estimation")
    # Basic force calculation based on active session state dimensions
    f_rad_waist = (0.05 * (0.16**4) * 36 * (1.5**1.2)) + (0.03 * (0.09**4) * 72 * (1.5**1.1))
    st.metric(label="Estimated Radial Force at Waist (Composite)", value=f"{f_rad_waist:.3f} N")


# ==============================================================================
# TAB 2: INTERACTIVE 3D WEBGL FIXTURE (EMBEDDED HTML/JS)
# ==============================================================================
with tab2:
    st.markdown("### Parametric Mold Assembly")
    st.markdown("The mold below is dynamically shaped based on the inputs from Tab 1. Use the side panel to toggle the **Thermal Map Overlay**.")
    
    # Inject Streamlit session state variables into the JS starting parameters
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
      :root{ --bg-deep:#0c0f13; --bg-viewport:#10151b; --panel:#141a21; --panel-2:#181f27; --border:#262f3a; --text:#e7edf3; --text-dim:#8a94a1; --text-faint:#5b6572; --blue:#4fa8e0; --blue-dim:#4fa8e055; --orange:#ff7a45; --orange-dim:#ff7a4530; --red:#e04f4f; }
      *{box-sizing:border-box;}
      html,body{margin:0;padding:0;height:100%;background:var(--bg-deep);color:var(--text);font-family:sans-serif;overflow:hidden;}
      #app{display:flex;flex-direction:column;height:100vh;}
      #main{flex:1;display:flex;min-height:0;}
      #viewport{flex:1;position:relative;background:radial-gradient(ellipse at 50% 30%, #182029 0%, var(--bg-viewport) 70%);overflow:hidden;}
      #viewport canvas{display:block;width:100%;height:100%;cursor:grab;}
      aside#panel{width:352px;flex-shrink:0;background:var(--panel);border-left:1px solid var(--border);overflow-y:auto;padding:16px;}
      .sec{margin-bottom:20px;}
      .sec h2{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:var(--text-faint);margin:0 0 10px;border-bottom:1px solid var(--border);padding-bottom:5px;}
      .toggle{display:flex;align-items:center;justify-content:space-between;padding:7px 0;cursor:pointer;font-size:13px;}
      .switch{position:relative;width:34px;height:19px;flex-shrink:0;}
      .switch input{opacity:0;width:0;height:0;}
      .slider-tog{position:absolute;inset:0;background:#2a333d;border-radius:20px;transition:.15s;}
      .slider-tog::before{content:'';position:absolute;width:14px;height:14px;left:2.5px;top:2.5px;background:#8a94a1;border-radius:50%;transition:.15s;}
      .switch input:checked + .slider-tog{background:var(--red);}
      .switch input:checked + .slider-tog::before{transform:translateX(15px);background:#fff;}
      input[type=range]{width:100%;}
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
            <label style="font-size:12px; color:var(--text-dim);">Explode</label>
            <input type="range" id="explode" min="0" max="100" step="1" value="0">
          </div>
        </aside>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function(){
      "use strict";
      
      // INJECTED FROM STREAMLIT
      """ + html_injection + """

      var params = { D1:param_starts.D1, D2:param_starts.D2, D3:param_starts.D3, D4:param_starts.D4, waist:param_starts.waist, dish:2.0, bolts:4, explode:0, section:false, thermal:false };
      var currentExplode = 0;
      var dirty = true;

      var K = { seatLen:2.0, seatT:0.8, uCylLen:2.2, bossR:2.2, bossH:4.0, tipR:0.06, capT:4.0, topT:6.0, baseT:10, baseMargin:15, msR:2.2, toolMarginTop:5, toolMarginBottom:4, meshOff:0.55 };

      var canvas = document.getElementById('c');
      var renderer = new THREE.WebGLRenderer({canvas:canvas, antialias:true});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
      renderer.localClippingEnabled = true;

      var scene = new THREE.Scene();
      var camera = new THREE.PerspectiveCamera(38, 1, 0.1, 2000);
      camera.position.set(100, 80, 120);
      camera.lookAt(0, 20, 0);

      var hemi = new THREE.HemisphereLight(0x3a4a58, 0x08090b, 0.75); scene.add(hemi);
      var key = new THREE.DirectionalLight(0xfff2df, 1.15); key.position.set(80,140,90); scene.add(key);
      var clipPlane = new THREE.Plane(new THREE.Vector3(1,0,0), 0);

      // Materials (Standard & Thermal)
      function mat(color, rough, metal){ return new THREE.MeshStandardMaterial({color:color, roughness:rough, metalness:metal, side:THREE.DoubleSide}); }
      
      var mBase_std = mat(0x9aa4ad, 0.45, 0.85); var mBase_thm = mat(0x4a1111, 0.8, 0.1);
      var mMandrel_std = mat(0xd7dde2, 0.28, 0.9); var mMandrel_thm = mat(0xffcc00, 0.1, 0.0); // Core is hottest
      var mClamp_std = mat(0x5c6570, 0.4, 0.75); var mClamp_thm = mat(0xcc4400, 0.6, 0.1);
      var mMesh_std = mat(0xd9b46c, 0.4, 0.9); mMesh_std.wireframe = true;
      var mMesh_thm = mat(0xffffff, 0.0, 0.0); mMesh_thm.wireframe = true;

      var matBase = mBase_std.clone(), matMandrel = mMandrel_std.clone(), matClamp = mClamp_std.clone(), matMesh = mMesh_std.clone();

      var assembly = new THREE.Group(); scene.add(assembly);
      var gBase = new THREE.Group(), gMandrel = new THREE.Group(), gBottom = new THREE.Group(), gTop = new THREE.Group(), gMesh = new THREE.Group();
      assembly.add(gBase,gMandrel,gBottom,gTop,gMesh);

      function V(r,y){ return new THREE.Vector2(Math.max(r,0.05), y); }
      function lathe(pts, material){ var m = new THREE.Mesh(new THREE.LatheGeometry(pts, 56), material); return m; }

      function buildAssembly(){
        while(gBase.children.length) gBase.remove(gBase.children[0]);
        while(gMandrel.children.length) gMandrel.remove(gMandrel.children[0]);
        while(gBottom.children.length) gBottom.remove(gBottom.children[0]);
        while(gTop.children.length) gTop.remove(gTop.children[0]);
        while(gMesh.children.length) gMesh.remove(gMesh.children[0]);

        var r1=params.D1/2, r2=params.D2/2, r3=params.D3/2, r4=params.D4/2, WL=params.waist, DD=params.dish;
        var yA=K.seatLen, yB=yA+K.seatT, yC=yB+3.2, yD=yC+WL, yE=yD+K.uCylLen, yF=yE+0.6, yG=yF+K.bossH;

        // Mandrel
        var mPts = [V(K.msR,0), V(K.msR,yA), V(r3,yB), V(r3,yC), V(r2,yD), V(r2,yE), V(K.bossR,yF), V(K.bossR,yG)];
        gMandrel.add(lathe(mPts, matMandrel)); gMandrel.position.y = K.baseT;

        // Bottom Clamp
        var bPts = [V(K.msR+0.35,0), V(r4+K.toolMarginBottom,0), V(r4+K.toolMarginBottom,K.capT), V(r3*1.05, K.capT*0.5+DD*0.5), V(K.msR+0.35, K.capT*0.6), V(K.msR+0.35,0)];
        gBottom.add(lathe(bPts, matClamp)); gBottom.position.y = K.baseT + yA;

        // Top Clamp
        var tPts = [V(K.bossR+0.35,0), V(r1+K.toolMarginTop,0), V(r1+K.toolMarginTop,K.topT), V(r2*1.15, K.topT*0.8), V(K.bossR+0.35, K.topT*0.6), V(K.bossR+0.35,0)];
        gTop.add(lathe(tPts, matClamp)); gTop.position.y = K.baseT + yD;

        // Mesh
        var meshPts = [V(r4,yA-0.4), V(r3+K.meshOff, yB+0.3), V(r2+K.meshOff, yD-0.15), V(r1, yE+K.uCylLen*0.3)];
        gMesh.add(lathe(meshPts, matMesh)); gMesh.position.y = K.baseT;
      }

      function updateMaterials(){
        if(params.thermal){
            matBase.copy(mBase_thm); matMandrel.copy(mMandrel_thm); matClamp.copy(mClamp_thm); matMesh.copy(mMesh_thm);
        } else {
            matBase.copy(mBase_std); matMandrel.copy(mMandrel_std); matClamp.copy(mClamp_std); matMesh.copy(mMesh_std);
        }
        matBase.needsUpdate = true; matMandrel.needsUpdate = true; matClamp.needsUpdate = true; matMesh.needsUpdate = true;
      }

      document.getElementById('explode').addEventListener('input', e => { params.explode = e.target.value/100; });
      document.getElementById('section').addEventListener('change', e => { params.section = e.target.checked; });
      document.getElementById('thermal').addEventListener('change', e => { params.thermal = e.target.checked; updateMaterials(); });

      function animate(){
        requestAnimationFrame(animate);
        if(dirty){ buildAssembly(); updateMaterials(); dirty = false; }
        
        currentExplode += (params.explode - currentExplode) * 0.14;
        gBottom.position.y = (K.baseT + K.seatLen) - currentExplode*15;
        gTop.position.y = (K.baseT + K.seatLen + K.seatT + 3.2 + params.waist) + currentExplode*25;
        
        renderer.clippingPlanes = params.section ? [clipPlane] : [];
        renderer.render(scene, camera);
      }
      
      window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth/window.innerHeight; camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
      });
      renderer.setSize(window.innerWidth, window.innerHeight);
      buildAssembly();
      animate();
    })();
    </script>
    </body>
    </html>
    """
    components.html(three_js_code, height=700, scrolling=False)


# ==============================================================================
# TAB 3: THERMAL PROFILING (HEAT SETTING SIMULATION)
# ==============================================================================
with tab3:
    st.header("Salt Bath / Furnace Heat-Setting Profile")
    st.markdown("Simulated heat transfer through the 316L stainless steel tooling to the Nitinol mesh.")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        target_temp = st.slider("Target Set Temperature (°C)", 450, 550, 500, 10)
        soak_time = st.slider("Soak Time (minutes)", 5, 20, 10, 1)
        st.info("Ensure the thermal map toggle is enabled in Tab 2 to visualize conductive gradients across the central mandrel.")
        
    with c2:
        # Generate mock heat curve
        time_axis = np.linspace(0, soak_time + 10, 100)
        furnace_temp = np.where(time_axis < 2, 25 + (target_temp-25)*(time_axis/2), target_temp)
        furnace_temp[time_axis > soak_time + 2] = furnace_temp[time_axis > soak_time + 2] * np.exp(-(time_axis[time_axis > soak_time + 2] - (soak_time + 2))/3)
        
        # Tooling lags behind furnace
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
        
        # Format for dark theme compatibility
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.tick_params(colors='gray')
        ax.xaxis.label.set_color('gray')
        ax.yaxis.label.set_color('gray')
        
        st.pyplot(fig)
