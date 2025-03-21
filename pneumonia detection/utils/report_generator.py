import random
from datetime import datetime

def generate_report(class_prediction, severity_prediction, class_probabilities):
    """Generate an AI-powered medical report based on the model's predictions.
    
    Args:
        class_prediction: Predicted class (Normal, Bacterial Pneumonia, Viral Pneumonia)
        severity_prediction: Predicted severity level (Mild, Moderate, Severe)
        class_probabilities: Dictionary containing probabilities for each class
        
    Returns:
        report: A formatted medical report as a string
    """
    # Get current date and time for the report
    current_datetime = datetime.now().strftime("%B %d, %Y %H:%M")
    
    # Initialize report sections
    report_header = f"RADIOLOGY REPORT\nDate: {current_datetime}\nExam: Chest X-Ray\n\n"
    
    # Clinical findings based on prediction
    findings = "FINDINGS:\n"
    
    if class_prediction == "Normal":
        findings += "The lungs are clear without evidence of focal consolidation, pneumonia, or pleural effusion. "
        findings += "Heart size and pulmonary vascularity appear within normal limits. "
        findings += "No pneumothorax or pneumomediastinum. "
        findings += "Osseous structures are unremarkable."
    else:
        # Common findings for pneumonia
        findings += "The examination reveals "
        
        # Specific findings based on type and severity
        if class_prediction == "Bacterial Pneumonia":
            if severity_prediction == "Mild":
                findings += "subtle patchy opacity in the lower lung field, suggestive of early bacterial pneumonia. "
                findings += "Heart size is normal. No pleural effusion is identified."
            elif severity_prediction == "Moderate":
                findings += "focal consolidation in the lower lung zone with air bronchograms, consistent with moderate bacterial pneumonia. "
                findings += "Mild blunting of the costophrenic angle may represent small pleural effusion."
            else:  # Severe
                findings += "extensive consolidation involving multiple lobes with air bronchograms, characteristic of severe bacterial pneumonia. "
                findings += "Associated moderate pleural effusion is noted. Heart size appears mildly enlarged."
        else:  # Viral Pneumonia
            if severity_prediction == "Mild":
                findings += "subtle bilateral interstitial opacities, more pronounced in the lower lung fields, suggestive of mild viral pneumonia. "
                findings += "No pleural effusion. Heart size is normal."
            elif severity_prediction == "Moderate":
                findings += "bilateral ground-glass opacities and interstitial pattern, consistent with moderate viral pneumonia. "
                findings += "No significant pleural effusion. Heart size is at the upper limits of normal."
            else:  # Severe
                findings += "diffuse bilateral ground-glass opacities and consolidations, indicating severe viral pneumonia. "
                findings += "Possible small pleural effusion cannot be excluded. Heart size appears mildly enlarged."
    
    # Impression section
    impression = "\nIMPRESSION:\n"
    
    if class_prediction == "Normal":
        impression += "No radiographic evidence of acute cardiopulmonary disease."
    else:
        confidence_level = ""
        max_prob = max(class_probabilities.values())
        
        if max_prob > 90:
            confidence_level = "high confidence"
        elif max_prob > 75:
            confidence_level = "moderate confidence"
        else:
            confidence_level = "low confidence"
        
        impression += f"Radiographic findings consistent with {severity_prediction.lower()} {class_prediction.lower()} (diagnostic confidence: {confidence_level})."
        
        # Recommendations based on severity
        impression += "\n\nRECOMMENDATIONS:\n"
        
        if severity_prediction == "Mild":
            impression += "1. Clinical correlation and appropriate antimicrobial therapy based on clinical presentation.\n"
            impression += "2. Follow-up imaging in 2-3 weeks to ensure resolution if clinically indicated."
        elif severity_prediction == "Moderate":
            impression += "1. Prompt initiation of appropriate antimicrobial therapy.\n"
            impression += "2. Consider additional laboratory tests including blood cultures and sputum analysis.\n"
            impression += "3. Follow-up imaging in 1-2 weeks to assess treatment response."
        else:  # Severe
            impression += "1. Urgent clinical assessment and appropriate antimicrobial therapy.\n"
            impression += "2. Consider hospital admission for close monitoring and supportive care.\n"
            impression += "3. Additional imaging (CT chest) may provide further characterization if clinically indicated.\n"
            impression += "4. Follow-up imaging in 24-48 hours to assess for rapid progression."
    
    # Combine all sections
    report = report_header + findings + impression
    
    return report