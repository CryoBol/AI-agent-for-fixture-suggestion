import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="PDA Occluder Heat-Setting Fixture Suite", layout="wide")

st.title("🔬 PDA Occluder Heat-Setting Fixture Studio")
st.markdown("Integrated computational platform featuring a dual-layer mechanics solver and an interactive WebGL parametric mold assembly.")

# Multi-Tab Layout
tab1, tab2 = st.tabs(["1. Dual-Layer Mechanics Engine", "2. Interactive 3D Fixture Assembly (WebGL)"])

# ==============================================================================
# TAB 1: DUAL-LAYER MECHANICS & FORCE SIMULATION
# ==============================================================================
with tab1:
    st.header("Dual-Layer Occluder Structural & Force Modeller")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("Macro Geometry & Inner Core")
        D_macro = st.slider("Waist Diameter (D_waist mm)", 4.0, 30.0, 8.0, 0.5, key="t1_D")
        P_macro = st.slider("Pitch Length (mm)", 1.0, 10.0, 4.0, 0.2, key="t1_P")
        N1 = st.slider("Inner Wire Count (N1)", 16, 72, 36, 4, key="t1_N1")
        d1 = st.slider("Inner Wire Diameter (d1 mm)", 0.10, 0.30, 0.16, 0.01, key="t1_d1")

    with col_t2:
        st.subheader("Outer Shell & Displacement")
        N2 = st.slider("Outer Wire Count (N2)", 36, 144, 72, 4, key="t1_N2")
        d2 = st.slider("Outer Wire Diameter (d2 mm)", 0.05, 0.15, 0.09, 0.01, key="t1_d2")
        delta_r = st.slider("Radial Displacement (delta_r mm)", 0.1, 5.0, 1.5, 0.1, key="t1_dr")

    def calculate_braid_geometry(D_mm, P_mm, N, d):
        tan_theta = (np.pi * D_mm) / P_mm
        theta_rad = np.arctan(tan_theta)
        cf = ((N * d) / (np.pi * D_mm * np.sin(theta_rad))) * (2.0 - ((N * d) / (np.pi * D_mm * np.sin(theta_rad))))
        return np.degrees(theta_rad), np.clip(cf, 0.0, 1.0)

    theta1, cf1 = calculate_braid_geometry(D_macro, P_macro, N1, d1)
    theta2, cf2 = calculate_braid_geometry(D_macro, P_macro, N2, d2)

    E_eff = 50000.0  # N/mm^2
    ei1 = (N1 * E_eff * np.pi * (d1**4) / 64.0) * (np.cos(np.radians(theta1))**2)
    ei2 = (N2 * E_eff * np.pi * (d2**4) / 64.0) * (np.cos(np.radians(theta2))**2)
    
    k1 = 0.05 * (d1**4) * N1
    k2 = 0.03 * (d2**4) * N2
    f_rad_waist = (k1 * (delta_r**1.2)) + (k2 * (delta_r**1.1))

    m1, m2, m3 = st.columns(3)
    m1.metric("Inner Braid Angle", f"{theta1:.2f}°")
    m2.metric("Outer Braid Angle", f"{theta2:.2f}°")
    m3.metric("Waist Radial Force", f"{f_rad_waist:.3f} N")

    st.subheader("Radial Force vs. Displacement Curve")
    fig1, ax1 = plt.subplots(figsize=(10, 3))
    displacements = np.linspace(0.1, 5.0, 50)
    forces = [(k1 * (d**1.2)) + (k2 * (d**1.1)) for d in displacements]
    ax1.plot(displacements, forces, label="Dual-Layer Composite Force", color="#008080", linewidth=2.5)
    ax1.axhline(y=0.1, color="orange", linestyle="--", label="Min Clinical Limit")
    ax1.set_xlabel("Radial Displacement (mm)")
    ax1.set_ylabel("Radial Force (N)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

# ==============================================================================
# TAB 2: INTERACTIVE 3D WEBGL FIXTURE (EMBEDDED HTML/JS)
# ==============================================================================
with tab2:
    st.markdown("### Parametric Mold Assembly (Reference CAD · rev A)")
    st.markdown("Use the integrated side panel to adjust the Aortic/Pulmonary dimensions, trigger the exploded view, or toggle the section cutaway.")
    
    # The exact HTML/Three.js code provided by the user
    three_js_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDA Occluder Heat-Setting Fixture — Parametric Mold Assembly</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      :root{
        --bg-deep:#0c0f13;
        --bg-viewport:#10151b;
        --panel:#141a21;
        --panel-2:#181f27;
        --border:#262f3a;
        --text:#e7edf3;
        --text-dim:#8a94a1;
        --text-faint:#5b6572;
        --blue:#4fa8e0;
        --blue-dim:#4fa8e055;
        --orange:#ff7a45;
        --orange-dim:#ff7a4530;
        --steel:#9aa4ad;
        --steel-dark:#5c6570;
        --brass:#c99a4e;
        --blackox:#22262c;
        --mesh:#d9b46c;
        --good:#5fbf8a;
      }
      *{box-sizing:border-box;}
      html,body{margin:0;padding:0;height:100%;background:var(--bg-deep);color:var(--text);font-family:'Space Grotesk',sans-serif;overflow:hidden;}
      #app{display:flex;flex-direction:column;height:100vh;}

      header#topbar{
        display:flex;align-items:baseline;gap:14px;
        padding:14px 20px;border-bottom:1px solid var(--border);
        background:linear-gradient(180deg,var(--panel-2),var(--panel));
        flex-shrink:0;z-index:5;
      }
      header#topbar h1{font-size:16px;font-weight:600;letter-spacing:.2px;margin:0;}
      header#topbar h1 .tag{color:var(--blue);font-weight:500;}
      header#topbar p{margin:0;font-size:12px;color:var(--text-dim);font-weight:400;}
      header#topbar .spacer{flex:1;}
      header#topbar .badge{
        font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--text-faint);
        border:1px solid var(--border);padding:3px 8px;border-radius:20px;letter-spacing:.5px;
      }

      #main{flex:1;display:flex;min-height:0;}
      #viewport{flex:1;position:relative;background:
          radial-gradient(ellipse at 50% 30%, #182029 0%, var(--bg-viewport) 70%);
        overflow:hidden;}
      #viewport canvas{display:block;width:100%;height:100%;cursor:grab;}
      #viewport canvas:active{cursor:grabbing;}
      #vignette{position:absolute;inset:0;pointer-events:none;
        box-shadow:inset 0 0 140px 40px rgba(0,0,0,0.55);}

      #hint{position:absolute;left:16px;bottom:14px;font-family:'JetBrains Mono',monospace;
        font-size:10.5px;color:var(--text-faint);letter-spacing:.3px;pointer-events:none;}

      #readout{position:absolute;left:16px;top:14px;background:rgba(20,26,33,0.82);
        border:1px solid var(--border);border-radius:8px;padding:10px 14px;
        font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text-dim);
        line-height:1.85;backdrop-filter:blur(4px);pointer-events:none;min-width:190px;}
      #readout b{color:var(--text);font-weight:600;}
      #readout .k{color:var(--text-faint);}
      #readout .hl{color:var(--orange);}

      #dimSvg{position:absolute;inset:0;pointer-events:none;width:100%;height:100%;}
      #dimSvg .dline{stroke:var(--blue);stroke-width:1.2;stroke-dasharray:4 3;opacity:.85;}
      #dimSvg .dtick{stroke:var(--blue);stroke-width:1.2;opacity:.85;}
      #dimSvg text{fill:var(--blue);font-family:'JetBrains Mono',monospace;font-size:11px;}
      #dimSvg .dbg{fill:#0c0f13;opacity:0.72;}

      aside#panel{width:352px;flex-shrink:0;background:var(--panel);border-left:1px solid var(--border);
        overflow-y:auto;padding:16px 16px 28px;}
      aside#panel::-webkit-scrollbar{width:8px;}
      aside#panel::-webkit-scrollbar-thumb{background:#2a333d;border-radius:4px;}

      .sec{margin-bottom:20px;}
      .sec h2{font-size:11px;text-transform:uppercase;letter-spacing:1.4px;color:var(--text-faint);
        margin:0 0 10px;font-weight:600;display:flex;align-items:center;gap:8px;}
      .sec h2::after{content:'';flex:1;height:1px;background:var(--border);}

      .row{display:flex;align-items:center;justify-content:space-between;margin-bottom:9px;gap:10px;}
      .row label{font-size:12.5px;color:var(--text);flex-shrink:0;}
      .row .val{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--blue);min-width:52px;text-align:right;}
      input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:#2a333d;border-radius:2px;outline:none;}
      input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;
        background:var(--orange);cursor:pointer;border:2px solid #1a1005;box-shadow:0 0 0 2px rgba(255,122,69,.18);}
      input[type=range]::-moz-range-thumb{width:13px;height:13px;border-radius:50%;background:var(--orange);border:2px solid #1a1005;cursor:pointer;}
      .slidewrap{display:flex;flex-direction:column;gap:4px;margin-bottom:12px;}

      .btnrow{display:flex;gap:6px;}
      .btnrow button{flex:1;background:var(--panel-2);border:1px solid var(--border);color:var(--text-dim);
        padding:6px 0;border-radius:6px;font-family:'JetBrains Mono',monospace;font-size:12px;cursor:pointer;transition:.15s;}
      .btnrow button.active{background:var(--orange-dim);border-color:var(--orange);color:var(--orange);}
      .btnrow button:hover{color:var(--text);}

      .toggle{display:flex;align-items:center;justify-content:space-between;padding:7px 0;cursor:pointer;}
      .toggle span{font-size:12.5px;color:var(--text);}
      .switch{position:relative;width:34px;height:19px;flex-shrink:0;}
      .switch input{opacity:0;width:0;height:0;}
      .slider-tog{position:absolute;inset:0;background:#2a333d;border-radius:20px;transition:.15s;}
      .slider-tog::before{content:'';position:absolute;width:14px;height:14px;left:2.5px;top:2.5px;background:#8a94a1;
        border-radius:50%;transition:.15s;}
      .switch input:checked + .slider-tog{background:var(--blue-dim);}
      .switch input:checked + .slider-tog::before{transform:translateX(15px);background:var(--blue);}

      .legend{display:flex;flex-direction:column;gap:2px;}
      .legrow{display:flex;align-items:center;gap:9px;padding:6px 4px;border-radius:6px;cursor:pointer;}
      .legrow:hover{background:var(--panel-2);}
      .swatch{width:11px;height:11px;border-radius:3px;flex-shrink:0;border:1px solid rgba(255,255,255,.15);}
      .legrow span{font-size:12px;color:var(--text-dim);flex:1;}
      .legrow input{accent-color:var(--blue);width:14px;height:14px;flex-shrink:0;}

      .note{font-size:11px;color:var(--text-faint);line-height:1.6;border-top:1px solid var(--border);
        padding-top:12px;margin-top:4px;}
      .note b{color:var(--text-dim);}

      #resetBtn{width:100%;background:var(--panel-2);border:1px solid var(--border);color:var(--text-dim);
        padding:9px 0;border-radius:7px;font-family:'Space Grotesk';font-size:12.5px;cursor:pointer;margin-top:4px;}
      #resetBtn:hover{color:var(--text);border-color:var(--blue);}

      @media (max-width:860px){
        #main{flex-direction:column;}
        aside#panel{width:100%;height:46vh;border-left:none;border-top:1px solid var(--border);}
        #viewport{height:54vh;}
      }
    </style>
    </head>
    <body>
    <div id="app">
      <header id="topbar">
        <h1>PDA Occluder <span class="tag">Heat-Setting Fixture</span></h1>
        <p>Parametric Nitinol mold assembly — mandrel · disc clamps · spacer · tie-rods</p>
        <div class="spacer"></div>
        <div class="badge">REFERENCE CAD · rev A</div>
      </header>
      <div id="main">
        <div id="viewport">
          <canvas id="c"></canvas>
          <svg id="dimSvg"></svg>
          <div id="vignette"></div>
          <div id="readout"></div>
          <div id="hint">drag — rotate &nbsp;·&nbsp; scroll — zoom &nbsp;·&nbsp; shift+drag — pan</div>
        </div>
        <aside id="panel">

          <div class="sec">
            <h2>Device Geometry</h2>
            <div class="slidewrap">
              <div class="row"><label>D1 — Aortic disc Ø</label><span class="val" id="v_d1">24.0 mm</span></div>
              <input type="range" id="d1" min="16" max="34" step="0.5" value="24">
            </div>
            <div class="slidewrap">
              <div class="row"><label>D2 — Aortic waist Ø</label><span class="val" id="v_d2">9.0 mm</span></div>
              <input type="range" id="d2" min="5" max="16" step="0.5" value="9">
            </div>
            <div class="slidewrap">
              <div class="row"><label>D3 — Pulmonary waist Ø</label><span class="val" id="v_d3">6.0 mm</span></div>
              <input type="range" id="d3" min="3" max="14" step="0.5" value="6">
            </div>
            <div class="slidewrap">
              <div class="row"><label>D4 — Pulmonary disc Ø</label><span class="val" id="v_d4">10.0 mm</span></div>
              <input type="range" id="d4" min="4" max="24" step="0.5" value="10">
            </div>
            <div class="slidewrap">
              <div class="row"><label>Waist length (cone)</label><span class="val" id="v_wl">9.0 mm</span></div>
              <input type="range" id="wl" min="4" max="20" step="0.5" value="9">
            </div>
            <div class="slidewrap">
              <div class="row"><label>Disc dish depth</label><span class="val" id="v_dd">2.0 mm</span></div>
              <input type="range" id="dd" min="0.5" max="4.5" step="0.1" value="2.0">
            </div>
          </div>

          <div class="sec">
            <h2>Fixture Setup</h2>
            <div class="row"><label>Loading screws</label></div>
            <div class="btnrow" style="margin-bottom:12px;">
              <button data-bolts="3" id="bolt3">3</button>
              <button data-bolts="4" id="bolt4" class="active">4</button>
              <button data-bolts="6" id="bolt6">6</button>
            </div>
          </div>

          <div class="sec">
            <h2>View</h2>
            <div class="slidewrap">
              <div class="row"><label>Explode assembly</label><span class="val" id="v_ex">0%</span></div>
              <input type="range" id="explode" min="0" max="100" step="1" value="0">
            </div>
            <label class="toggle"><span>Section view (cutaway)</span>
              <span class="switch"><input type="checkbox" id="section"><span class="slider-tog"></span></span>
            </label>
            <label class="toggle"><span>Show dimensions</span>
              <span class="switch"><input type="checkbox" id="showDims" checked><span class="slider-tog"></span></span>
            </label>
            <button id="resetBtn">Reset view</button>
          </div>

          <div class="sec">
            <h2>Components</h2>
            <div class="legend" id="legend">
              <label class="legrow"><span class="swatch" style="background:#9aa4ad"></span><span>Base plate / jig frame</span><input type="checkbox" id="ly_base" checked></label>
              <label class="legrow"><span class="swatch" style="background:#cfd6db"></span><span>Central mandrel / core pin</span><input type="checkbox" id="ly_mandrel" checked></label>
              <label class="legrow"><span class="swatch" style="background:#5c6570"></span><span>Pulmonary disc clamp (bottom)</span><input type="checkbox" id="ly_bottom" checked></label>
              <label class="legrow"><span class="swatch" style="background:#c99a4e"></span><span>Spacer / positioning ring</span><input type="checkbox" id="ly_spacer" checked></label>
              <label class="legrow"><span class="swatch" style="background:#5c6570"></span><span>Aortic disc clamp (top)</span><input type="checkbox" id="ly_top" checked></label>
              <label class="legrow"><span class="swatch" style="background:#22262c"></span><span>Loading screws / clips</span><input type="checkbox" id="ly_bolts" checked></label>
              <label class="legrow"><span class="swatch" style="background:#d9b46c"></span><span>Nitinol mesh (reference)</span><input type="checkbox" id="ly_mesh" checked></label>
            </div>
          </div>

          <div class="note"><b>Note —</b> this is a parametric reference model for design review and communication, not a certified manufacturing drawing. Verify all dimensions, tolerances, and material call-outs against the device IFU and your fixture drawing package before machining.</div>
        </aside>
      </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function(){
      "use strict";

      // ---------- state ----------
      var params = { D1:24, D2:9, D3:6, D4:10, waist:9, dish:2.0, bolts:4, explode:0, section:false, showDims:true };
      var layers = { base:true, mandrel:true, bottom:true, spacer:true, top:true, bolts:true, mesh:true };
      var dirty = true;
      var currentExplode = 0;

      // fixed tooling constants (mm)
      var K = {
        seatLen:2.0, seatT:0.8, uCylLen:2.2, bossR:2.2, bossH:4.0, tipR:0.06,
        capT:4.0, topT:6.0, baseT:10, baseMargin:15, msR:2.2,
        toolMarginTop:5, toolMarginBottom:4, meshOff:0.55,
        boltShaftR:1.5, boltHeadR:3.0, boltHeadH:3.0, nutR:3.3, nutH:4.5,
        liftBottom:12, liftSpacer:24, liftTop:42, liftBolt:60, fanBolt:9
      };

      // ---------- three.js setup ----------
      var viewportEl = document.getElementById('viewport');
      var canvas = document.getElementById('c');
      var renderer = new THREE.WebGLRenderer({canvas:canvas, antialias:true});
      renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      if(THREE.sRGBEncoding) renderer.outputEncoding = THREE.sRGBEncoding;
      renderer.localClippingEnabled = true;

      var scene = new THREE.Scene();
      scene.background = new THREE.Color(0x10151b);
      scene.fog = new THREE.Fog(0x10151b, 220, 520);

      var camera = new THREE.PerspectiveCamera(38, 1, 0.1, 2000);

      // lights
      var hemi = new THREE.HemisphereLight(0x3a4a58, 0x08090b, 0.75);
      scene.add(hemi);
      var key = new THREE.DirectionalLight(0xfff2df, 1.15);
      key.position.set(80,140,90);
      key.castShadow = true;
      key.shadow.mapSize.set(1024,1024);
      key.shadow.camera.left=-90;key.shadow.camera.right=90;key.shadow.camera.top=90;key.shadow.camera.bottom=-90;
      key.shadow.camera.far=400;
      scene.add(key);
      var rim = new THREE.DirectionalLight(0x4fa8e0, 0.45);
      rim.position.set(-100,60,-80);
      scene.add(rim);
      var fill = new THREE.PointLight(0xffffff, 0.25);
      fill.position.set(-40,40,60);
      scene.add(fill);

      // floor / grid
      var floor = new THREE.Mesh(
        new THREE.CircleGeometry(220,64),
        new THREE.MeshStandardMaterial({color:0x0d1116, roughness:1, metalness:0})
      );
      floor.rotation.x = -Math.PI/2;
      floor.position.y = -0.05;
      floor.receiveShadow = true;
      scene.add(floor);
      var grid = new THREE.GridHelper(300, 60, 0x2a5878, 0x1a222b);
      grid.position.y = -0.02;
      grid.material.opacity = 0.35;
      grid.material.transparent = true;
      scene.add(grid);

      // clipping plane for section view
      var clipPlane = new THREE.Plane(new THREE.Vector3(1,0,0), 0);

      // ---------- materials ----------
      function mat(color, rough, metal, extra){
        var o = {color:color, roughness:rough, metalness:metal, side:THREE.DoubleSide};
        if(extra) for(var k in extra) o[k]=extra[k];
        return new THREE.MeshStandardMaterial(o);
      }
      var matBase   = mat(0x9aa4ad, 0.45, 0.85);
      var matMandrel= mat(0xd7dde2, 0.28, 0.9);
      var matClamp  = mat(0x5c6570, 0.4, 0.75);
      var matSpacer = mat(0xc99a4e, 0.35, 0.7);
      var matBolt   = mat(0x22262c, 0.5, 0.6);

      function makeHatchTexture(){
        var size=256;
        var c=document.createElement('canvas'); c.width=size;c.height=size;
        var ctx=c.getContext('2d');
        ctx.clearRect(0,0,size,size);
        ctx.strokeStyle='rgba(255,255,255,0.95)';
        ctx.lineWidth=3.2;
        var step=30;
        for(var i=-size;i<size*2;i+=step){
          ctx.beginPath();ctx.moveTo(i,0);ctx.lineTo(i+size,size);ctx.stroke();
          ctx.beginPath();ctx.moveTo(i,size);ctx.lineTo(i-size,size);ctx.stroke();
        }
        var tex=new THREE.CanvasTexture(c);
        tex.wrapS=THREE.RepeatWrapping; tex.wrapT=THREE.RepeatWrapping;
        tex.repeat.set(14,4);
        return tex;
      }
      var matMesh = new THREE.MeshBasicMaterial({
        map:makeHatchTexture(), color:0xd9b46c, transparent:true, opacity:0.82,
        side:THREE.DoubleSide, depthWrite:false
      });

      // ---------- groups ----------
      var assembly = new THREE.Group(); scene.add(assembly);
      var gBase = new THREE.Group(); var gMandrel = new THREE.Group();
      var gBottom = new THREE.Group(); var gSpacer = new THREE.Group();
      var gTop = new THREE.Group(); var gMesh = new THREE.Group();
      var gBolts = new THREE.Group();
      assembly.add(gBase,gMandrel,gBottom,gSpacer,gTop,gMesh,gBolts);

      var baseY = { bottom:0, spacer:0, top:0 };
      var boltDefs = []; // {angle, r}
      var dims = {}; // label defs

      function disposeGroup(g){
        for(var i=g.children.length-1;i>=0;i--){
          var ch = g.children[i];
          if(ch.geometry) ch.geometry.dispose();
          g.remove(ch);
        }
      }
      function V(r,y){ return new THREE.Vector2(Math.max(r,0.05), y); }

      function latheMesh(pts, material, segs){
        var geo = new THREE.LatheGeometry(pts, segs||56);
        geo.computeVertexNormals();
        var m = new THREE.Mesh(geo, material);
        m.castShadow = true; m.receiveShadow = true;
        return m;
      }

      // ---------- core build ----------
      function buildAssembly(){
        // clamp coupled dimensions
        if(params.D2 <= params.D3 + 1) params.D2 = params.D3 + 1;
        if(params.D4 <= params.D3 + 1) params.D4 = params.D3 + 1;
        syncSliderFromParam('d2', params.D2);
        syncSliderFromParam('d4', params.D4);

        disposeGroup(gBase); disposeGroup(gMandrel); disposeGroup(gBottom);
        disposeGroup(gSpacer); disposeGroup(gTop); disposeGroup(gMesh); disposeGroup(gBolts);

        var D1=params.D1, D2=params.D2, D3=params.D3, D4=params.D4, WL=params.waist, DD=params.dish;
        var r1=D1/2, r2=D2/2, r3=D3/2, r4=D4/2;

        // ---- local Y levels along the mandrel (0 = base-plate top surface) ----
        var yA = K.seatLen;                              // top of mount pin
        var yB = yA + K.seatT;                            // start of pulmonary straight tube
        var yC = yB + 3.2;                                 // top of pulmonary straight tube (spacer/cap zone)
        var yD = yC + WL;                                  // top of taper = D2 (aortic) plane
        var yE = yD + K.uCylLen;                           // top of aortic straight tube
        var yF = yE + 0.6;                                 // step to boss
        var yG = yF + K.bossH;                             // top of locating boss

        // ---- base plate (ring, world y 0..baseT) ----
        var baseOuterR = r1 + K.baseMargin;
        var basePts = [ V(K.msR,0), V(baseOuterR,0), V(baseOuterR,K.baseT), V(K.msR,K.baseT), V(K.msR,0) ];
        var baseMesh = latheMesh(basePts, matBase, 72);
        baseMesh.position.y = 0;
        gBase.add(baseMesh);
        gBase.visible = layers.base;

        // ---- central mandrel ----
        var mPts = [
          V(K.msR, 0), V(K.msR, yA),
          V(r3, yB), V(r3, yC),
          V(r2, yD), V(r2, yE),
          V(K.bossR, yF), V(K.bossR, yG),
          V(K.tipR, yG+0.35)
        ];
        var mandrelMesh = latheMesh(mPts, matMandrel, 56);
        gMandrel.add(mandrelMesh);
        gMandrel.position.y = K.baseT;
        gMandrel.visible = layers.mandrel;

        // ---- bottom end cap : pulmonary-side disc-forming die ----
        var bOuter = r4 + K.toolMarginBottom;
        var bore1 = K.msR + 0.35;
        var bPts = [
          V(bore1, 0), V(bOuter, 0),
          V(bOuter, K.capT*0.82),
          V(r3*1.05, K.capT*0.5 + DD*0.5),
          V(bore1+1.4, K.capT*0.42),
          V(bore1, K.capT*0.6),
          V(bore1, 0)
        ];
        var bottomMesh = latheMesh(bPts, matClamp, 56);
        baseY.bottom = K.baseT + yA;
        gBottom.add(bottomMesh);
        gBottom.position.y = baseY.bottom;
        gBottom.visible = layers.bottom;

        // ---- spacer / positioning ring ----
        var spOuter = r3 + 3.0;
        var spBore = r3 + 0.35;
        var spH = Math.max(yC-yB, 1.5);
        var spPts = [ V(spBore,0), V(spOuter,0), V(spOuter,spH), V(spBore,spH), V(spBore,0) ];
        var spacerMesh = latheMesh(spPts, matSpacer, 48);
        baseY.spacer = K.baseT + yB;
        gSpacer.add(spacerMesh);
        gSpacer.position.y = baseY.spacer;
        gSpacer.visible = layers.spacer;

        // ---- top clamp : aortic-side disc-forming die ----
        var tOuter = r1 + K.toolMarginTop;
        var tBore = K.bossR + 0.35;
        var tPts = [
          V(tBore, 0), V(tOuter, 0),
          V(tOuter, K.topT),
          V(r2*1.15, K.topT*0.82),
          V(r1*0.88, K.topT*0.55 + DD),
          V(tBore+1.6, K.topT*0.5),
          V(tBore, K.topT*0.66),
          V(tBore, 0)
        ];
        var topMesh = latheMesh(tPts, matClamp, 56);
        baseY.top = K.baseT + yD;
        gTop.add(topMesh);
        gTop.position.y = baseY.top;
        gTop.visible = layers.top;

        // ---- nitinol mesh overlay (reference envelope) ----
        var off = K.meshOff;
        var meshPts = [
          V(r4, yA-0.4), V(r3+off, yB+0.3), V(r3+off, yC-0.3),
          V(r2+off, yD-0.15), V(r2+off, yE+0.1), V(r1, yE + K.uCylLen*0.3 + DD*0.9)
        ];
        var meshShell = latheMesh(meshPts, matMesh, 64);
        gMesh.add(meshShell);
        gMesh.position.y = K.baseT;
        gMesh.visible = layers.mesh;

        // ---- loading screws / bolts ----
        var boltCircleR = r1 + 9.5;
        var totalTop = baseY.top + K.topT; // assembled top surface
        var shaftLen = totalTop + K.boltHeadH + K.nutH + 2;
        boltDefs = [];
        var n = params.bolts;
        for(var i=0;i<n;i++){
          var ang = (i/n)*Math.PI*2 + Math.PI/n*0.3;
          boltDefs.push({angle:ang, r:boltCircleR});
          var g = new THREE.Group();
          var shaft = new THREE.Mesh(new THREE.CylinderGeometry(K.boltShaftR,K.boltShaftR,shaftLen,20), matBolt);
          shaft.position.y = shaftLen/2 - K.boltHeadH;
          shaft.castShadow = true;
          var head = new THREE.Mesh(new THREE.CylinderGeometry(K.boltHeadR,K.boltHeadR,K.boltHeadH,6), matBolt);
          head.position.y = -K.boltHeadH/2;
          head.castShadow = true;
          var nutY = totalTop + K.boltHeadH + K.nutH/2;
          var nut = new THREE.Mesh(new THREE.CylinderGeometry(K.nutR,K.nutR,K.nutH,8), matBolt);
          nut.position.y = nutY;
          nut.castShadow = true;
          g.add(shaft, head, nut);
          g.position.set(Math.cos(ang)*boltCircleR, -K.boltHeadH, Math.sin(ang)*boltCircleR);
          g.userData.baseAngle = ang; g.userData.baseR = boltCircleR;
          gBolts.add(g);
        }
        gBolts.visible = layers.bolts;

        // ---- dimension label anchors (assembled Y, before explode) ----
        dims = {
          D1: { r:r1, y: baseY.top + K.topT*0.55 + DD, lift:'top', value:D1, label:'D1' },
          D2: { r:r2, y: K.baseT + yD, lift:'none', value:D2, label:'D2' },
          D3: { r:r3, y: K.baseT + (yB+yC)/2, lift:'none', value:D3, label:'D3' },
          D4: { r:r4, y: baseY.bottom + K.capT*0.5, lift:'bottom', value:D4, label:'D4' }
        };

        // camera framing target (first build only)
        var totalHeight = totalTop;
        if(!camReady){
          camTarget.set(0, totalHeight*0.42, 0);
          camRadius = Math.max(90, totalHeight*2.1);
          camReady = true;
          updateCamera();
        }

        // readout
        var halfAngleDeg = Math.atan((r2-r3)/WL) * 180/Math.PI;
        document.getElementById('readout').innerHTML =
          '<span class="k">Cone half-angle</span> &nbsp; <b class="hl">'+halfAngleDeg.toFixed(1)+'&deg;</b><br>'+
          '<span class="k">Fixture height</span> &nbsp; <b>'+totalHeight.toFixed(1)+' mm</b><br>'+
          '<span class="k">Bolt circle Ø</span> &nbsp; <b>'+(boltCircleR*2).toFixed(1)+' mm</b><br>'+
          '<span class="k">Base plate Ø</span> &nbsp; <b>'+(baseOuterR*2).toFixed(1)+' mm</b>';
      }

      function syncSliderFromParam(id, val){
        var el = document.getElementById(id);
        if(Math.abs(parseFloat(el.value)-val) > 0.01){ el.value = val; }
      }

      // ---------- explode + section application (every frame) ----------
      function applyExplode(t){
        gBottom.position.y = baseY.bottom + t*K.liftBottom;
        gSpacer.position.y = baseY.spacer + t*K.liftSpacer;
        gTop.position.y    = baseY.top    + t*K.liftTop;

        for(var i=0;i<gBolts.children.length;i++){
          var g = gBolts.children[i];
          var ang = g.userData.baseAngle, r = g.userData.baseR + t*K.fanBolt;
          g.position.x = Math.cos(ang)*r;
          g.position.z = Math.sin(ang)*r;
          g.position.y = -K.boltHeadH + t*K.liftBolt;
        }

        var meshT = Math.max(0, 1 - t/0.4);
        gMesh.visible = layers.mesh && meshT > 0.02;
        matMesh.opacity = 0.82*meshT;

        renderer.clippingPlanes = params.section ? [clipPlane] : [];
      }

      // ---------- dimension overlay (SVG) ----------
      var svg = document.getElementById('dimSvg');
      function updateDims(){
        if(!params.showDims){ svg.innerHTML=''; return; }
        var w = viewportEl.clientWidth, h = viewportEl.clientHeight;
        svg.setAttribute('viewBox','0 0 '+w+' '+h);
        var html = '';
        var liftMap = { none:0, bottom: currentExplode*K.liftBottom, top: currentExplode*K.liftTop };
        for(var key2 in dims){
          var d = dims[key2];
          if(key2==='D1' && currentExplode>0.55) continue;
          if(key2==='D4' && currentExplode>0.55) continue;
          var y = d.y + liftMap[d.lift];
          var p1 = toScreen(new THREE.Vector3(-d.r,y,0));
          var p2 = toScreen(new THREE.Vector3( d.r,y,0));
          if(!p1.visible || !p2.visible) continue;
          var mx=(p1.x+p2.x)/2, my=(p1.y+p2.y)/2;
          html += '<line class="dline" x1="'+p1.x+'" y1="'+p1.y+'" x2="'+p2.x+'" y2="'+p2.y+'"/>';
          html += '<line class="dtick" x1="'+p1.x+'" y1="'+(p1.y-5)+'" x2="'+p1.x+'" y2="'+(p1.y+5)+'"/>';
          html += '<line class="dtick" x1="'+p2.x+'" y1="'+(p2.y-5)+'" x2="'+p2.x+'" y2="'+(p2.y+5)+'"/>';
          var lbl = d.label+' '+d.value.toFixed(1)+'mm';
          html += '<rect class="dbg" x="'+(mx-30)+'" y="'+(my-15)+'" width="60" height="15" rx="3"></rect>';
          html += '<text x="'+mx+'" y="'+(my-4)+'" text-anchor="middle">'+lbl+'</text>';
        }
        svg.innerHTML = html;
      }
      function toScreen(v3){
        var v = v3.clone().project(camera);
        var w = viewportEl.clientWidth, h = viewportEl.clientHeight;
        return { x:(v.x*0.5+0.5)*w, y:(-v.y*0.5+0.5)*h, visible: v.z<1 };
      }

      // ---------- camera controls (custom orbit) ----------
      var camTheta = Math.PI*0.28, camPhi = Math.PI*0.36, camRadius = 140;
      var camTarget = new THREE.Vector3(0,20,0);
      var camReady = false;
      function updateCamera(){
        var x = camTarget.x + camRadius*Math.sin(camPhi)*Math.sin(camTheta);
        var y = camTarget.y + camRadius*Math.cos(camPhi);
        var z = camTarget.z + camRadius*Math.sin(camPhi)*Math.cos(camTheta);
        camera.position.set(x,y,z);
        camera.lookAt(camTarget);
      }
      var dragging=false, panning=false, lastX=0, lastY=0;
      canvas.addEventListener('contextmenu', function(e){ e.preventDefault(); });
      canvas.addEventListener('pointerdown', function(e){
        if(e.button===2 || e.shiftKey) panning=true; else dragging=true;
        lastX=e.clientX; lastY=e.clientY;
        canvas.setPointerCapture(e.pointerId);
      });
      window.addEventListener('pointerup', function(){ dragging=false; panning=false; });
      window.addEventListener('pointermove', function(e){
        var dx=e.clientX-lastX, dy=e.clientY-lastY;
        lastX=e.clientX; lastY=e.clientY;
        if(dragging){
          camTheta -= dx*0.0062;
          camPhi -= dy*0.0062;
          camPhi = Math.max(0.12, Math.min(Math.PI-0.12, camPhi));
          updateCamera();
        } else if(panning){
          var dist = camRadius, speed = dist*0.0013;
          var fwd = new THREE.Vector3(); camera.getWorldDirection(fwd);
          var right = new THREE.Vector3().crossVectors(fwd, camera.up).normalize();
          var up = new THREE.Vector3().crossVectors(right, fwd).normalize();
          camTarget.addScaledVector(right, -dx*speed);
          camTarget.addScaledVector(up, dy*speed);
          updateCamera();
        }
      });
      canvas.addEventListener('wheel', function(e){
        e.preventDefault();
        camRadius *= (1 + e.deltaY*0.0011);
        camRadius = Math.max(25, Math.min(480, camRadius));
        updateCamera();
      }, {passive:false});
      document.getElementById('resetBtn').addEventListener('click', function(){
        camTheta = Math.PI*0.28; camPhi = Math.PI*0.36;
        camReady = false; dirty = true;
        updateCamera();
      });

      // ---------- resize ----------
      function onResize(){
        var w = viewportEl.clientWidth, h = viewportEl.clientHeight;
        renderer.setSize(w,h,false);
        camera.aspect = w/h;
        camera.updateProjectionMatrix();
      }
      window.addEventListener('resize', onResize);

      // ---------- UI wiring ----------
      function bindSlider(id, key3, unit, decimals){
        var el = document.getElementById(id), out = document.getElementById('v_'+id);
        el.addEventListener('input', function(){
          params[key3] = parseFloat(el.value);
          out.textContent = params[key3].toFixed(decimals)+' '+unit;
          dirty = true;
        });
        out.textContent = params[key3].toFixed(decimals)+' '+unit;
      }
      bindSlider('d1','D1','mm',1);
      bindSlider('d2','D2','mm',1);
      bindSlider('d3','D3','mm',1);
      bindSlider('d4','D4','mm',1);
      bindSlider('wl','waist','mm',1);
      bindSlider('dd','dish','mm',1);

      var explodeEl = document.getElementById('explode'), explodeOut = document.getElementById('v_ex');
      explodeEl.addEventListener('input', function(){
        params.explode = parseInt(explodeEl.value,10)/100;
        explodeOut.textContent = explodeEl.value+'%';
      });

      document.getElementById('section').addEventListener('change', function(e){ params.section = e.target.checked; });
      document.getElementById('showDims').addEventListener('change', function(e){ params.showDims = e.target.checked; });

      ['bolt3','bolt4','bolt6'].forEach(function(id){
        document.getElementById(id).addEventListener('click', function(){
          params.bolts = parseInt(this.getAttribute('data-bolts'),10);
          ['bolt3','bolt4','bolt6'].forEach(function(i){ document.getElementById(i).classList.remove('active'); });
          this.classList.add('active');
          dirty = true;
        });
      });

      var layerMap = { ly_base:'base', ly_mandrel:'mandrel', ly_bottom:'bottom', ly_spacer:'spacer', ly_top:'top', ly_bolts:'bolts', ly_mesh:'mesh' };
      var layerGroups = { base:gBase, mandrel:gMandrel, bottom:gBottom, spacer:gSpacer, top:gTop, bolts:gBolts, mesh:gMesh };
      Object.keys(layerMap).forEach(function(id){
        document.getElementById(id).addEventListener('change', function(e){
          var k = layerMap[id];
          layers[k] = e.target.checked;
          layerGroups[k].visible = layers[k];
        });
      });

      // ---------- main loop ----------
      function animate(){
        requestAnimationFrame(animate);
        if(dirty){ buildAssembly(); dirty = false; }
        currentExplode += (params.explode - currentExplode) * 0.14;
        if(Math.abs(currentExplode-params.explode) < 0.001) currentExplode = params.explode;
        applyExplode(currentExplode);
        updateDims();
        renderer.render(scene, camera);
      }

      onResize();
      updateCamera();
      buildAssembly();
      dirty = false;
      animate();

      window.addEventListener('load', onResize);
    })();
    </script>
    </body>
    </html>
    """

    # Embed the custom Three.js app inside Streamlit, setting the height to accommodate the panel
    components.html(three_js_code, height=850, scrolling=False)
