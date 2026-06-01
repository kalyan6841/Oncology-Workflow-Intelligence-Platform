from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import altair as alt


def patient_interface_dashboard() -> None:
    st.header("Patient Companion Interface")

    # Patient selection
    patient_options = list(st.session_state.patients_df["patient_id"].astype(str).unique())
    selected_patient_id = st.selectbox("Select Patient ID", patient_options)

    if selected_patient_id:
        patient_data = st.session_state.patients_df[st.session_state.patients_df["patient_id"] == selected_patient_id].iloc[0]

        # Treatment Summary
        st.subheader("Treatment Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Name:** {patient_data['name']}")
            st.write(f"**Age:** {patient_data['age']}")
            st.write(f"**Gender:** {patient_data['gender']}")
            st.write(f"**Cancer Type:** {patient_data['cancer_type']}")
            st.write(f"**Stage:** {patient_data['stage']}")
        with col2:
            st.write(f"**Regimen Sequence:** {patient_data['regimen_sequence']}")
            st.write(f"**Current Cycle:** {patient_data['cycle_number']}")
            st.write(f"**Treatment Intent:** {patient_data['treatment_intent']}")
            st.write(f"**Frequency:** Every {patient_data['frequency_days']} days")

        # Cycle Readiness Indicator
        st.subheader("Cycle Readiness Indicator")
        lab_status = patient_data['lab_status']
        pap_status = patient_data['pap_status']
        if lab_status == 'Ready' and pap_status == 'Ready':
            st.success("✅ Ready for next cycle")
        elif lab_status == 'Pending' or pap_status == 'Pending':
            st.warning("⚠️ Pending lab or PAP results")
        else:
            st.error("❌ Not ready - check with coordinator")

        # Lab Reminders
        st.subheader("Lab Reminders")
        if lab_status == 'Pending':
            st.info("📅 Upcoming lab tests required before next cycle")
        else:
            st.success("✅ Lab tests completed")

        # Doctor Review Reminders
        st.subheader("Doctor Review Reminders")
        if pap_status == 'Pending':
            st.info("📅 PAP smear or doctor review pending")
        else:
            st.success("✅ Doctor review completed")

        # Chemo Admission Alerts
        st.subheader("Chemo Admission Alerts")
        last_cycle_date = pd.to_datetime(patient_data['last_cycle_date'])
        frequency_days = int(patient_data['frequency_days'])
        next_cycle_date = last_cycle_date + timedelta(days=frequency_days)
        days_until_next = (next_cycle_date.date() - datetime.now().date()).days
        if days_until_next <= 3:
            st.warning(f"🚨 Admission scheduled in {days_until_next} days ({next_cycle_date.strftime('%Y-%m-%d')})")
        else:
            st.info(f"Next admission: {next_cycle_date.strftime('%Y-%m-%d')} ({days_until_next} days)")

        # Treatment Timeline Visualization
        st.subheader("Treatment Timeline Visualization")
        # Build timeline data
        timeline_data = []
        current_date = pd.to_datetime(patient_data['last_cycle_date'])
        cycle_number = int(patient_data['cycle_number'])
        for cycle in range(1, cycle_number + 5):  # Show current and next few cycles
            timeline_data.append({
                'Cycle': f'Cycle {cycle}',
                'Date': current_date.strftime('%Y-%m-%d'),
                'Type': 'Completed' if cycle <= cycle_number else 'Upcoming'
            })
            current_date += timedelta(days=frequency_days)

        timeline_df = pd.DataFrame(timeline_data)
        timeline_df['Date'] = pd.to_datetime(timeline_df['Date'])

        chart = alt.Chart(timeline_df).mark_circle(size=100).encode(
            x='Date:T',
            y='Cycle:N',
            color='Type:N',
            tooltip=['Cycle', 'Date', 'Type']
        ).properties(width=600, height=300)
        st.altair_chart(chart, use_container_width=True)

        # Complaint Submission Panel
        st.subheader("Complaint Submission Panel")
        with st.form("complaint_form"):
            complaint_type = st.selectbox("Complaint Type", ["Treatment side effects", "Scheduling issues", "Staff behavior", "Facility issues", "Other"])
            complaint_description = st.text_area("Description")
            submitted = st.form_submit_button("Submit Complaint")
            if submitted:
                if complaint_description.strip():
                    complaint = {
                        "patient_id": selected_patient_id,
                        "complaint_type": complaint_type,
                        "description": complaint_description,
                        "submitted_at": str(datetime.now()),
                        "status": "Submitted"
                    }
                    if 'complaints' not in st.session_state:
                        st.session_state.complaints = []
                    st.session_state.complaints.append(complaint)
                    st.success("Complaint submitted successfully. It will appear in the Complaint Escalation Panel.")
                else:
                    st.warning("Please provide a description for the complaint.")