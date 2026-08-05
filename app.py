import streamlit as st
import numpy as np
import plotly.graph_objects as go
import cadquery as cq

st.set_page_config(page_title="PDA Occluder Core Fixture AI", layout="wide")

st.title("🔬 Precision PDA Occluder Core & Mandrel Fixture")
st.markdown("Restoring the open assembly fixture format with a precision central shaping mandrel matching the exact asymmetric PDA occluder profile.")

# --- SIDEBAR: USER INPUTS (PDA DEVICE PARAMETERS) ---
st.sidebar.header("1. PDA Occluder Target Dimensions")

pulm_dia = st.sidebar.slider("Pulmonary End Diameter ($D_{pulm}$)", 4.0, 10.0, 6.0, step=0.5)
waist_dia = st.sidebar.slider("Central Waist Diameter ($D_{waist}$)", 2.0, 8.0, 4.0, step=0.5)
aortic_dia = st.sidebar.slider("Aortic Retention Disc Diameter ($D_{aortic}$)", 12.0, 30.0, 16.0, step=0.5)
length = st.sidebar.slider("Total Device Length ($L$)", 8.0, 25.0, 14.0, step=0.5)
wire_count = st.sidebar.slider("Nitinol Wire Ends", 36, 144, 72, step=12)

st.sidebar.subheader("Heat Treatment Parameters")
temp = st.sidebar.number_input("Target Setting Temperature (°C)", 400, 600, 500)
soak_time = st.sidebar.number_input("Soak Time (mins)", 5, 60, 15)

# --- OPEN ASSEMBLY CORE FIXTURE CAD GENERATION ---
def generate_pda_core_fixture(d_pulm, d_waist, d_aortic, lg, wires):
    center_bore_r = 3.0
    pulm_r = max(d_pulm / 2, center_bore_r + 1.0)
    waist_r = max(d_waist / 2, center_bore_r + 0.8)
    aortic_r = max(d_aortic / 2, waist_r + 2.0)
    
    plate_r = aortic_r + 12.0
    base_r = plate_r + 10.0
    
    assy = cq.Assembly(name="PDA_Core_Fixture_Assembly")

    # 1. Base Stand
    base = (cq.Workplane("XY")
            .circle(base_r).extrude(12)
            .faces(">Z").chamfer(1.5)
            .faces(">Z").workplane()
            .circle(plate_r).extrude(6)
            .faces(">Z").workplane().circle(center_bore_r).cutThruAll())
    assy.add(base, name="BaseStand", color=cq.Color(0.55, 0.55, 0.58))

    # 2. Bottom Retention Plate
    bottom_plate = (cq.Workplane("XY").circle(plate_r).extrude(4)
                    .faces(">Z").workplane().circle(center_bore_r).cutThruAll())
    bottom_plate = bottom_plate.faces(">Z").workplane().polarArray(plate_r - 4, 0, 360, 4).circle(1.6).cutThruAll()
    bottom_plate = bottom_plate.faces(">Z").workplane().polarArray(plate_r - 4, 45, 360, 2).circle(2.0).cutThruAll()
    
    for r_idx in range(2):
        curr_r = center_bore_r + 2.0 + (r_idx * ((pulm_r - center_bore_r - 2) / max(1, 2)))
        bottom_plate = bottom_plate.faces(">Z").workplane().polarArray(curr_r, 0, 360, 24).circle(0.35).cutThruAll()

    assy.add(bottom_plate, name="BottomRetentionPlate", loc=cq.Location(cq.Vector(0, 0, 18)), color=cq.Color(0.8, 0.8, 0.85))

    # 3. Top Retention Plate
    top_plate = (cq.Workplane("XY").circle(plate_r).extrude(4)
                 .faces(">Z").workplane().circle(center_bore_r).cutThruAll())
    top_plate = top_plate.faces(">Z").workplane().polarArray(plate_r - 4, 0, 360, 4).circle(1.6).cutThruAll()
    top_plate = top_plate.faces(">Z").workplane().polarArray(plate_r - 4, 45, 360, 2).circle(2.0).cutThruAll()
    
    for r_idx in range(3):
        curr_r = center_bore_r + 2.0 + (r_idx * ((aortic_r - center_bore_r - 2) / max(1, 3)))
        top_plate = top_plate.faces(">Z").workplane().polarArray(curr_r, 0, 360, 36).circle(0.35).cutThruAll()

    assy.add(top_plate, name="TopRetentionPlate", loc=cq.Location(cq.Vector(0, 0, 18 + 4 + lg)), color=cq.Color(0.8, 0.8, 0.85))

    # 4. Central Solid PDA Shaping Core (Mandrel) positioned between plates
    core_profile = (cq.Workplane("XZ")
                    .moveTo(center_bore_r, 0)
                    .lineTo(pulm_r, 0)
                    .threePointArc((waist_r, lg * 0.4), (aortic_r, lg))
                    .lineTo(center_bore_r, lg)
                    .close().revolve(360, (0, 0, 0), (0, 0, 1)))

    assy.add(core_profile, name="PDAShapingCore", loc=cq.Location(cq.Vector(0, 0, 22)), color=cq.Color(0.9, 0.6, 0.1))

    # 5. Fasteners and Tie Rods
    stud = cq.Workplane("XY").circle(1.5).extrude(lg + 28)
    nut = cq.Workplane("XY").polygon(6, 4.5).extrude(3).faces(">Z").workplane().circle(1.5).cutThruAll()
    pin = cq.Workplane("XY").circle(1.9).extrude(lg + 16)

    for i in range(4):
        angle = np.radians(i * 90)
        x, y = (plate_r - 4) * np.cos(angle), (plate_r - 4) * np.sin(angle)
        assy.add(stud, name=f"Bolt_{i}", loc=cq.Location(cq.Vector(x, y, 10)), color=cq.Color(0.7, 0.7, 0.7))
        assy.add(nut, name=f"BottomNut_{i}", loc=cq.Location(cq.Vector(x, y, 18)), color=cq.Color(0.75, 0.75, 0.75))
        assy.add(nut, name=f"TopNut_{i}", loc=cq.Location(cq.Vector(x, y, 18 + 4 + lg + 4)), color=cq.Color(0.75, 0.75, 0.75))

    for i in range(2):
        angle = np.radians(i * 180 + 45)
        x, y = (plate_r - 4) * np.cos(angle), (plate_r - 4) * np.sin(angle)
        assy.add(pin, name=f"AlignPin_{i}", loc=cq.Location(cq.Vector(x, y, 18)), color=cq.Color(0.4, 0.4, 0.4))

    return assy


# --- TRIGGER EVALUATION ---
if st.button("🚀 Generate Open-Core PDA Fixture & Render Thermal Profile", type="primary", use_container_width=True):
    
    with st.spinner("Computing open-assembly CAD geometry and mapping thermal gradients..."):
        try:
            assy = generate_pda_core_fixture(pulm_dia, waist_dia, aortic_dia, length, wire_count)
            compound = assy.toCompound()
            
            vertices, triangles = compound.tessellate(0.2)
            
            if not vertices or not triangles:
                raise ValueError("Generated geometry resulted in an empty mesh.")

            x = np.array([v.x for v in vertices])
            y = np.array([v.y for v in vertices])
            z = np.array([v.z for v in vertices])
            i = np.array([t[0] for t in triangles])
            j = np.array([t[1] for t in triangles])
            k = np.array([t[2] for t in triangles])
            
            z_mid = 22 + (length / 2)
            dist_from_center = np.sqrt(x**2 + y**2)
            T = temp - (np.abs(z - z_mid) * 0.4) - (dist_from_center * 0.15)
            
            st.success("Open-Core PDA Fixture & Thermal Field Computed Successfully.")
            
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    intensity=T,
                    colorscale='Inferno',
                    colorbar=dict(title="Temperature (°C)", len=0.75),
                    flatshading=True,
                    showscale=True
                )
            ])
            
            fig.update_layout(
                title=f"Open-Assembly PDA Shaping Core Fixture (Target: {temp}°C)",
                scene=dict(
                    xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
                    aspectmode="data"
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=700
            )
            st.plotly_chart(fig, use_container_width=True)
            
            filename = "PDA_Core_Mandrel_Fixture.step"
            cq.exporters.export(compound, filename)
            
            with open(filename, "rb") as file:
                st.download_button(
                    label="💾 Download Fixture Manufacturing CAD (.STEP)",
                    data=file,
                    file_name=filename,
                    mime="application/step",
                    type="primary",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"Engine Exception: {str(e)}")