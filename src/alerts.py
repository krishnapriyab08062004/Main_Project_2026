"""
Alert Notification System
Send SMS and Email alerts when emotions exceed thresholds

To enable actual SMS/Email sending:
1. SMS: Sign up for Twilio (https://www.twilio.com/)
2. Email: Use SendGrid, AWS SES, or Gmail SMTP
3. Add your credentials to the .env file
"""

import os
from datetime import datetime
from typing import Optional

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Using environment variables only.")

# ============================================================================
# SMS ALERTS (Twilio)
# ============================================================================

def send_sms_twilio(phone: str, message: str) -> bool:
    """
    Send SMS using Twilio
    
    Setup:
    1. Sign up at https://www.twilio.com/
    2. Get Account SID, Auth Token, and Phone Number
    3. Add to .env file:
       TWILIO_ACCOUNT_SID=your_account_sid
       TWILIO_AUTH_TOKEN=your_auth_token
       TWILIO_PHONE_NUMBER=your_twilio_number
    
    Args:
        phone (str): Recipient phone number (format: +1234567890)
        message (str): Message to send
        
    Returns:
        bool: Success status
    """
    try:
        from twilio.rest import Client
        
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        from_number = os.getenv('TWILIO_PHONE_NUMBER')
        
        if not all([account_sid, auth_token, from_number]):
            print("⚠️ Twilio credentials not configured")
            return False
        
        client = Client(account_sid, auth_token)
        
        message = client.messages.create(
            body=message,
            from_=from_number,
            to=phone
        )
        
        print(f"✓ SMS sent successfully (SID: {message.sid})")
        return True
        
    except ImportError:
        print("⚠️ Twilio library not installed. Run: pip install twilio")
        return False
    except Exception as e:
        print(f"❌ Error sending SMS: {str(e)}")
        return False

# ============================================================================
# EMAIL ALERTS
# ============================================================================

def send_email_sendgrid(to_email: str, subject: str, message: str) -> bool:
    """
    Send email using SendGrid
    
    Setup:
    1. Sign up at https://sendgrid.com/
    2. Create API key
    3. Add to .env file:
       SENDGRID_API_KEY=your_api_key
       SENDGRID_FROM_EMAIL=your_verified_sender@example.com
    
    Args:
        to_email (str): Recipient email
        subject (str): Email subject
        message (str): Email body
        
    Returns:
        bool: Success status
    """
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail
        
        api_key = os.getenv('SENDGRID_API_KEY')
        from_email = os.getenv('SENDGRID_FROM_EMAIL')
        
        if not all([api_key, from_email]):
            print("⚠️ SendGrid credentials not configured")
            return False
        
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject=subject,
            html_content=f"<p>{message}</p>"
        )
        
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        
        print(f"✓ Email sent successfully (Status: {response.status_code})")
        return True
        
    except ImportError:
        print("⚠️ SendGrid library not installed. Run: pip install sendgrid")
        return False
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
        return False

def send_email_smtp(to_email: str, subject: str, message: str) -> bool:
    """
    Send email using Gmail SMTP
    
    Setup:
    1. Enable 2-factor authentication on Gmail
    2. Generate App Password: https://myaccount.google.com/apppasswords
    3. Add to .env file:
       GMAIL_ADDRESS=your_email@gmail.com
       GMAIL_APP_PASSWORD=your_app_password
    
    Args:
        to_email (str): Recipient email
        subject (str): Email subject
        message (str): Email body
        
    Returns:
        bool: Success status
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        gmail_address = os.getenv('GMAIL_ADDRESS')
        gmail_password = os.getenv('GMAIL_APP_PASSWORD')
        
        if not all([gmail_address, gmail_password]):
            print("⚠️ Gmail credentials not configured")
            return False
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = gmail_address
        msg['To'] = to_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(gmail_address, gmail_password)
            server.send_message(msg)
        
        print(f"✓ Email sent successfully via Gmail SMTP")
        return True
        
    except Exception as e:
        print(f"❌ Error sending email: {str(e)}")
        return False

# ============================================================================
# UNIFIED ALERT FUNCTIONS
# ============================================================================

def send_emotion_alert_sms(phone: str, emotion: str, confidence: float) -> bool:
    """
    Send emotion alert via SMS
    
    Args:
        phone (str): Phone number
        emotion (str): Detected emotion
        confidence (float): Confidence score
        
    Returns:
        bool: Success status
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"""
🎭 EMOTION ALERT

Detected: {emotion.upper()}
Confidence: {confidence:.1%}
Time: {timestamp}

Speech Emotion Recognition System
    """.strip()
    
    return send_sms_twilio(phone, message)

def send_emotion_alert_email(email: str, emotion: str, confidence: float) -> bool:
    """
    Send emotion alert via Email
    
    Args:
        email (str): Email address
        emotion (str): Detected emotion
        confidence (float): Confidence score
        
    Returns:
        bool: Success status
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    subject = f"🎭 Emotion Alert: {emotion.upper()} Detected"
    
    message = f"""
Emotion Alert Notification

Details:
- Emotion: {emotion.upper()}
- Confidence: {confidence:.1%}
- Timestamp: {timestamp}

This alert was triggered because the confidence level exceeded the configured threshold.

Speech Emotion Recognition System
    """.strip()
    
    # Try SendGrid first, fallback to Gmail SMTP
    if send_email_sendgrid(email, subject, message):
        return True
    else:
        return send_email_smtp(email, subject, message)

# ============================================================================
# CONFIGURATION INSTRUCTIONS
# ============================================================================

def print_setup_instructions():
    """Print setup instructions for alert services"""
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                    ALERT NOTIFICATION SETUP                           ║
╚═══════════════════════════════════════════════════════════════════════╝

📱 SMS ALERTS (Twilio):
   1. Sign up: https://www.twilio.com/try-twilio
   2. Get your Account SID and Auth Token
   3. Get a Twilio phone number
   4. Create a .env file in project root with:
      
      TWILIO_ACCOUNT_SID=your_account_sid_here
      TWILIO_AUTH_TOKEN=your_auth_token_here
      TWILIO_PHONE_NUMBER=+1234567890
   
   5. Install: pip install twilio python-dotenv

📧 EMAIL ALERTS (Option 1 - SendGrid):
   1. Sign up: https://sendgrid.com/
   2. Create and verify sender email
   3. Generate API key
   4. Add to .env file:
      
      SENDGRID_API_KEY=your_api_key_here
      SENDGRID_FROM_EMAIL=verified@yourdomain.com
   
   5. Install: pip install sendgrid python-dotenv

📧 EMAIL ALERTS (Option 2 - Gmail SMTP):
   1. Enable 2-factor authentication on your Gmail
   2. Generate App Password: https://myaccount.google.com/apppasswords
   3. Add to .env file:
      
      GMAIL_ADDRESS=your_email@gmail.com
      GMAIL_APP_PASSWORD=your_16_char_app_password
   
   4. Install: pip install python-dotenv

💡 TIPS:
   - Keep your .env file secure and never commit it to Git
   - Add .env to your .gitignore file
   - Test your configuration before using in production
   - Monitor usage to avoid exceeding free tier limits

📝 Example .env file:
   
   # Twilio SMS
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=your_auth_token_here
   TWILIO_PHONE_NUMBER=+15551234567
   
   # SendGrid Email
   SENDGRID_API_KEY=SG.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   SENDGRID_FROM_EMAIL=alerts@yourdomain.com
   
   # OR Gmail SMTP
   GMAIL_ADDRESS=your.email@gmail.com
   GMAIL_APP_PASSWORD=abcd efgh ijkl mnop

═══════════════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    print_setup_instructions()
    
    # Test configuration
    print("\n" + "="*70)
    print("TESTING CONFIGURATION")
    print("="*70 + "\n")
    
    # Test SMS
    test_phone = os.getenv('TEST_PHONE_NUMBER', '+1234567890')
    print(f"Testing SMS to {test_phone}...")
    send_emotion_alert_sms(test_phone, 'angry', 0.85)
    
    # Test Email
    test_email = os.getenv('TEST_EMAIL', 'test@example.com')
    print(f"\nTesting Email to {test_email}...")
    send_emotion_alert_email(test_email, 'sad', 0.78)
    
    print("\n" + "="*70)
    print("If you see '⚠️' warnings, configure the services as shown above")
    print("="*70)
