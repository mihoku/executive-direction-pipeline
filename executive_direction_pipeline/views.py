import requests
import json
import os
import time  # Import time for potential delays between retries
from concurrent.futures import ThreadPoolExecutor
import tempfile

from django.shortcuts import get_object_or_404
from django.http import JsonResponse, HttpResponseServerError, Http404
from django.views import View
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction

# for handling file upload
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.storage import default_storage, FileSystemStorage
from django.conf import settings
from django.utils.text import get_valid_filename

# REST framework django
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser

# mimetypes for file handling
import mimetypes
mimetypes.add_type("audio/m4a", ".m4a") # Override default mapping

from .models import Unit, MinutesOfMeeting, TaskAssignment
from .serializers import UnitSerializer, MinutesOfMeetingSerializer, TaskAssignmentSerializer

import math
from pydub import AudioSegment
from huggingface_hub import InferenceClient
from docling.document_converter import DocumentConverter

from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env

# --- Configuration ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
LLM_MODEL_NAME = os.environ.get("OPENROUTER_MODEL")
LLM_MODEL_NAME_2 = os.environ.get("OPENROUTER_MODEL_2")
TRANSCRIPTION_MODEL = "openai/whisper-large-v3"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 5 # Maximum number of retries for LLM calls
CHUNK_LENGTH_MS = 600 * 1000  # 10 minutes audio chunking for transcription
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".flac", ".ogg", ".webm"}

# --- Helper Functions ---

def get_unit_reference_string():
    """Fetches all units and formats them for the LLM prompt."""
    units = Unit.objects.all()
    if not units:
        return "No organizational units found in the database. Cannot assign tasks."
    unit_list = "\n".join([f"- Abbreviation: {u.abbreviation}, Name: {u.name}, Description: {u.description}" for u in units])
    return f"Available Organizational Units for assignment:\n{unit_list}\n"

def call_llm(prompt, llm_model="openai/gpt-oss-20b", is_json_mode=True, attempt=1):
    """
    Calls the OpenRouter API with a given prompt.
    Handles potential errors and checks for valid JSON if requested.
    Includes simple exponential backoff for retries.
    """
    if not OPENROUTER_API_KEY:
        print("ERROR: OPENROUTER_API_KEY not found in environment variables.")
        return None, "API key not configured."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}]
    }
    if is_json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Exponential backoff: wait 1s, 2s, 4s, 8s... before retrying
    if attempt > 1:
        delay = 2**(attempt - 2) # Start with 1 sec delay on 2nd attempt
        print(f"LLM call attempt {attempt}. Waiting {delay}s before retrying...")
        time.sleep(delay)

    try:
        print(f"--- Calling LLM (Attempt {attempt}) ---")
        # print(f"Prompt: {prompt[:500]}...") # Log truncated prompt for debugging

        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=120) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

        if not content:
            return None, "LLM returned empty content."

        if is_json_mode:
            try:
                # Attempt to parse the content as JSON
                parsed_json = json.loads(content)
                print("--- LLM Response (JSON Parsed Successfully) ---")
                return parsed_json, None # Success
            except json.JSONDecodeError as e:
                print(f"ERROR: LLM response is not valid JSON: {e}")
                print(f"Raw Content: {content}")
                return None, f"LLM response is not valid JSON: {e}" # JSON error
        else:
             print("--- LLM Response (Text) ---")
             return content.strip(), None # Success for text response

    except requests.exceptions.Timeout:
        print("ERROR: LLM request timed out.")
        return None, "API request timed out."
    except requests.exceptions.RequestException as e:
        print(f"ERROR: LLM request failed: {e}")
        error_detail = f"API request failed: {e}"
        if e.response is not None:
             try:
                 error_content = e.response.json()
                 error_detail += f" - Server Response: {error_content}"
             except json.JSONDecodeError:
                 error_detail += f" - Server Response: {e.response.text}"
        return None, error_detail # Network or HTTP error
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during LLM call: {e}")
        return None, f"An unexpected error occurred: {e}"

def convert_pdf_to_markdown(uploaded_file: InMemoryUploadedFile):
    """ Saves PDF temporarily, converts using Docling, cleans up. """
    temp_file = None
    temp_file_path = None
    try:
        # Save PDF content to a temporary file
        original_filename = uploaded_file.name or ""
        suffix = os.path.splitext(original_filename)[1] or ".pdf" # Ensure .pdf suffix if none
        if suffix.lower() != ".pdf": # Basic check
             print(f"Warning: Non-PDF suffix '{suffix}' provided, attempting conversion anyway.")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file_path = temp_file.name
        print(f"--- Saving PDF to temporary file: {temp_file_path} ---")
        uploaded_file.seek(0) # Go to start of uploaded file
        temp_file.write(uploaded_file.read())
        temp_file.close() # Close file before passing path to docling

        print(f"--- Starting Docling PDF conversion for: {temp_file_path} ---")
        converter = DocumentConverter()
        # Use the temporary file path as the source
        doc = converter.convert(temp_file_path).document
        markdown_text = doc.export_to_markdown()
        print("--- Docling conversion successful ---")
        return markdown_text.strip(), None

    except Exception as e:
        print(f"ERROR: Docling PDF conversion failed: {e}")
        import traceback
        traceback.print_exc()
        # You might want to return more specific errors from docling if possible
        return None, f"PDF conversion failed: {e}"
    finally:
        # --- IMPORTANT: Clean up the temporary PDF file ---
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"--- Deleted temporary PDF file: {temp_file_path} ---")
            except Exception as cleanup_error:
                print(f"ERROR: Failed to delete temporary PDF file {temp_file_path}: {cleanup_error}")
        elif temp_file: # Fallback cleanup
             try:
                 if not temp_file.closed: temp_file.close()
                 if hasattr(temp_file, 'name') and os.path.exists(temp_file.name): os.remove(temp_file.name)
             except Exception as cleanup_error: print(f"ERROR during secondary PDF cleanup: {cleanup_error}")

def transcribe_audio(uploaded_file: InMemoryUploadedFile):
    """
    1. Converts any audio/video to MP3.
    2. Splits it into 30-second chunks.
    3. Transcribes each chunk.
    4. Merges the text.
    """
    if not HF_API_KEY:
        return None, "Hugging Face API key not configured."

    temp_input_path = None
    chunk_files = [] # Keep track of chunks to clean up
    
    try:
        # --- Step 1: Save Uploaded File ---
        original_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if original_ext not in ALLOWED_EXTENSIONS:
            original_ext = ".tmp"
            
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=original_ext)
        temp_input_path = temp_input.name
        
        uploaded_file.seek(0)
        temp_input.write(uploaded_file.read())
        temp_input.close()

        print(f"--- Processing File: {temp_input_path} ---")

        # --- Step 2: Load & Convert to Audio ---
        # AudioSegment handles mp4/m4a/etc automatically
        audio = AudioSegment.from_file(temp_input_path)
        
        full_text = []
        duration_ms = len(audio)
        chunks_count = math.ceil(duration_ms / CHUNK_LENGTH_MS)
        
        print(f"--- Total Duration: {duration_ms/1000:.2f}s | Chunks: {chunks_count} ---")

        client = InferenceClient(api_key=HF_API_KEY, provider="hf-inference")

        # --- Step 3: Loop Through Chunks ---
        for i in range(chunks_count):
            start_ms = i * CHUNK_LENGTH_MS
            end_ms = min((i + 1) * CHUNK_LENGTH_MS, duration_ms)
            
            # Extract chunk
            chunk_audio = audio[start_ms:end_ms]
            
            # Export chunk to temp mp3
            chunk_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            chunk_temp_path = chunk_temp.name
            chunk_temp.close()
            chunk_files.append(chunk_temp_path)
            
            chunk_audio.export(chunk_temp_path, format="mp3")
            
            print(f"--- Transcribing Chunk {i+1}/{chunks_count} ({start_ms/1000}s - {end_ms/1000}s) ---")
            
            # Retry logic for individual chunks
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    output = client.automatic_speech_recognition(
                        audio=chunk_temp_path,
                        model=TRANSCRIPTION_MODEL
                    )
                    
                    text_segment = ""
                    if isinstance(output, dict) and 'text' in output:
                        text_segment = output['text'].strip()
                    elif isinstance(output, str):
                        text_segment = output.strip()
                    
                    if text_segment:
                        full_text.append(text_segment)
                    break # Success, move to next chunk
                    
                except Exception as e:
                    print(f"WARNING: Chunk {i+1} failed (Attempt {attempt+1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        print("ERROR: Skipping chunk due to repeated failures.")

        # --- Step 4: Combine Results ---
        final_transcription = " ".join(full_text)
        print("--- Transcription Complete ---")
        return final_transcription, None

    except Exception as e:
        print(f"ERROR: {e}")
        return None, f"Transcription failed: {str(e)}"

    finally:
        # --- Cleanup ---
        # 1. Input file
        if temp_input_path and os.path.exists(temp_input_path):
            try: os.remove(temp_input_path)
            except: pass
            
        # 2. Chunk files
        for chunk_path in chunk_files:
            if os.path.exists(chunk_path):
                try: os.remove(chunk_path)
                except: pass

def transcribe_audio_backup(uploaded_file: InMemoryUploadedFile):
    """
    Saves the uploaded file temporarily, calls Hugging Face ASR API using the file path,
    and then deletes the temporary file.
    """
    if not HF_API_KEY:
        print("ERROR: HF_TOKEN not found for transcription.")
        return None, "Hugging Face API key not configured."

    # Use NamedTemporaryFile to handle creation and cleanup
    # 'delete=False' allows us to close it, pass the path, then delete manually
    temp_file = None
    try:
        # Create a temporary file with the correct suffix if possible
        # Get original extension, handle cases without extension
        original_filename = uploaded_file.name or ""
        suffix = os.path.splitext(original_filename)[1] or ".tmp" # Use .tmp if no extension

        # Create named temp file (gets a unique name)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file_path = temp_file.name

        print(f"--- Saving uploaded content to temporary file: {temp_file_path} ---")
        # Write the uploaded content to the temporary file in binary mode
        uploaded_file.seek(0) # Ensure reading from the start
        temp_file.write(uploaded_file.read())
        temp_file.close() # Close the file so the InferenceClient can read it

        print(f"--- Calling Hugging Face ASR (Whisper) with file path: {temp_file_path} ---")
        client = InferenceClient(api_key=HF_API_KEY, provider="hf-inference")

        # Call the ASR method with the FILE PATH
        output = client.automatic_speech_recognition(
            audio=temp_file_path, # Pass the path, not the content bytes
            model=TRANSCRIPTION_MODEL
        )

        print(f"--- Transcription Successful ---")

        # Process output (same as before)
        if isinstance(output, dict) and 'text' in output:
             return output['text'].strip(), None
        elif isinstance(output, str):
             return output.strip(), None
        else:
             print(f"ERROR: Unexpected transcription output format: {type(output)}, Content: {output}")
             return None, "Unexpected format from transcription API."

    except Exception as e:
        print(f"ERROR: Hugging Face ASR request failed: {e}")
        # Log details if possible
        if hasattr(e, 'response') and e.response is not None:
             print(f"Server Response Status: {e.response.status_code}")
             try: print(f"Server Response Body: {e.response.json()}")
             except json.JSONDecodeError: print(f"Server Response Body (non-JSON): {e.response.text}")
        import traceback
        traceback.print_exc()
        return None, f"Transcription failed: {e}"
    finally:
        # --- IMPORTANT: Clean up the temporary file ---
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"--- Deleted temporary file: {temp_file_path} ---")
            except Exception as cleanup_error:
                print(f"ERROR: Failed to delete temporary file {temp_file_path}: {cleanup_error}")
        elif temp_file: # Handle cases where path wasn't assigned but file object exists
             try:
                 # If temp_file wasn't closed, closing it might delete it if delete=True was used originally
                 if not temp_file.closed: temp_file.close()
                 # If delete=False was used and path failed, try removing using file object name attribute
                 if hasattr(temp_file, 'name') and os.path.exists(temp_file.name): os.remove(temp_file.name)
             except Exception as cleanup_error:
                 print(f"ERROR: Failed during secondary cleanup attempt: {cleanup_error}")

# --- API Views ---

class UnitListView(generics.ListAPIView):
    queryset = Unit.objects.all()
    serializer_class = UnitSerializer

class MinutesOfMeetingListCreateView(generics.ListCreateAPIView):
    queryset = MinutesOfMeeting.objects.order_by('-created_at')
    serializer_class = MinutesOfMeetingSerializer

class TranscribeAndCreateMomView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def _save_uploaded_file(self, mom_id, uploaded_file):
        """Saves the uploaded file to the media directory."""
        try:
            # Create a unique-ish directory for this MoM's uploads
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'mom_audio', str(mom_id))
            os.makedirs(upload_dir, exist_ok=True) # Ensure directory exists

            # Clean the filename to prevent directory traversal issues etc.
            filename = get_valid_filename(uploaded_file.name)
            filepath = os.path.join(upload_dir, filename)

            # Use default_storage to save the file
            # This handles potential filename conflicts automatically if needed
            saved_path = default_storage.save(filepath, uploaded_file)
            print(f"Successfully saved uploaded file to: {saved_path}")
            # Optionally return the relative path if you store it in the model
            # return os.path.relpath(saved_path, settings.MEDIA_ROOT)
            return saved_path # Return the full saved path for now
        except Exception as e:
            print(f"ERROR: Could not save uploaded file '{uploaded_file.name}': {e}")
            # Decide if this should be a critical error or just logged
            return None


    def post(self, request, *args, **kwargs):
        title = request.data.get('title')
        meeting_date = request.data.get('meeting_date')
        audio_file = request.FILES.get('audio_file')

        # Basic Validation (keep as before)
        if not title: return Response({"error": "Missing title."}, status=status.HTTP_400_BAD_REQUEST)
        if not meeting_date: return Response({"error": "Missing meeting_date."}, status=status.HTTP_400_BAD_REQUEST)
        if not audio_file: return Response({"error": "Missing audio_file."}, status=status.HTTP_400_BAD_REQUEST)

        print(f"Received file: {audio_file.name}, Size: {audio_file.size}, Type: {audio_file.content_type}")

        

        # Read file content for transcription (keep as before)
        try:
            # Need to read the content first for transcription,
            # then potentially rewind or use the original file object for saving
            audio_file.seek(0) # Ensure we read from the beginning
            audio_content = audio_file.read()
            audio_file.seek(0) # Rewind again in case saving needs the original object state
            if not audio_content:
                return Response({"error": "Uploaded file appears to be empty."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
             print(f"ERROR: Could not read uploaded file: {e}")
             return Response({"error": f"Could not read uploaded file: {e}"}, status=status.HTTP_400_BAD_REQUEST)

        # --- Call transcription service WITH content_type ---
        transcribed_text, error = transcribe_audio(audio_file)
        # ----------------------------------------------------

        if error:
            print(f"Transcription error for file {audio_file.name}: {error}")
            return Response({"error": f"Transcription failed."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if transcribed_text is None or not transcribed_text.strip():
             print(f"Transcription returned empty text for file {audio_file.name}.")
             return Response({"error": "Transcription resulted in empty text."}, status=status.HTTP_400_BAD_REQUEST)

        # Create MoM object with transcribed text (keep as before)
        try:
            mom = MinutesOfMeeting.objects.create(
                title=title,
                meeting_date=meeting_date,
                text=transcribed_text,
                original_audio_filename=audio_file.name
            )
            print(f"Successfully created MoM {mom.id} from transcription.")

            # --- Save the uploaded file AFTER MoM is created ---
            saved_file_path = self._save_uploaded_file(mom.id, audio_file)
            if not saved_file_path:
                # Decide how to handle save failure: maybe log it but still return success for MoM?
                # Or maybe delete the MoM and return an error? Let's just log for now.
                print(f"Warning: MoM {mom.id} created, but failed to save the associated audio file.")
            # ----------------------------------------------------

            serializer = MinutesOfMeetingSerializer(mom)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            print(f"ERROR: Failed to save transcribed MoM to database: {e}")
            return Response({"error": f"Failed to save meeting minutes after transcription: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ConvertPdfAndCreateMomView(APIView):
    """ Handles POST with PDF file upload for conversion and MoM creation. """
    parser_classes = (MultiPartParser, FormParser) # Enable file uploads

    def _save_uploaded_pdf(self, mom_id, uploaded_file):
        """Saves the original uploaded PDF file permanently."""
        try:
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'mom_pdf', str(mom_id))
            os.makedirs(upload_dir, exist_ok=True)
            filename = get_valid_filename(uploaded_file.name)
            filepath = os.path.join(upload_dir, filename)
            uploaded_file.seek(0) # Ensure reading from start for saving
            saved_path = default_storage.save(filepath, uploaded_file)
            print(f"Successfully saved uploaded PDF to: {saved_path}")
            return saved_path # Return the path relative to MEDIA_ROOT if needed later
        except Exception as e:
            print(f"ERROR: Could not save uploaded PDF '{uploaded_file.name}': {e}")
            return None

    def post(self, request, *args, **kwargs):
        title = request.data.get('title')
        meeting_date = request.data.get('meeting_date')
        pdf_file = request.FILES.get('pdf_file') # Match frontend name

        # Validation
        if not title: return Response({"error": "Missing title."}, status=status.HTTP_400_BAD_REQUEST)
        if not meeting_date: return Response({"error": "Missing meeting_date."}, status=status.HTTP_400_BAD_REQUEST)
        if not pdf_file: return Response({"error": "Missing pdf_file."}, status=status.HTTP_400_BAD_REQUEST)

        # Basic file validation (check extension)
        if not pdf_file.name.lower().endswith('.pdf'):
            return Response({"error": "Invalid file type. Only PDF is allowed."}, status=status.HTTP_400_BAD_REQUEST)

        print(f"Received PDF file: {pdf_file.name}, Size: {pdf_file.size}, Type: {pdf_file.content_type}")

        # Call PDF conversion helper (which uses temp file)
        markdown_text, error = convert_pdf_to_markdown(pdf_file)

        if error:
            print(f"PDF conversion error for file {pdf_file.name}: {error}")
            # Give a generic error message to the frontend
            return Response({"error": f"PDF Conversion failed. Please check the file format or server logs."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if markdown_text is None or not markdown_text.strip():
             print(f"PDF conversion returned empty text for file {pdf_file.name}.")
             return Response({"error": "PDF conversion resulted in empty text. The PDF might be image-based or empty."}, status=status.HTTP_400_BAD_REQUEST)

        # Create MoM object
        try:
            mom = MinutesOfMeeting.objects.create(
                title=title,
                meeting_date=meeting_date,
                text=markdown_text, # Save converted markdown
                original_pdf_filename=pdf_file.name # Save original filename
            )
            print(f"Successfully created MoM {mom.id} from PDF (File: {pdf_file.name}).")

            # Save the original PDF file permanently AFTER MoM is created
            saved_pdf_path = self._save_uploaded_pdf(mom.id, pdf_file)
            if not saved_pdf_path:
                # Log the warning but don't fail the request
                print(f"Warning: MoM {mom.id} created, but failed to save the original PDF file.")

            serializer = MinutesOfMeetingSerializer(mom)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            # Catch DB errors etc.
            print(f"ERROR: Failed to save converted PDF MoM to database: {e}")
            return Response({"error": f"Failed to save meeting minutes after PDF conversion: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class MinutesOfMeetingDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = MinutesOfMeeting.objects.all()
    serializer_class = MinutesOfMeetingSerializer

class GenerateSummaryView(APIView):
    def post(self, request, pk):
        mom = get_object_or_404(MinutesOfMeeting, pk=pk)
        prompt = f"""
        Please provide a concise executive summary of the following minutes of meeting.
        Focus on the key decisions, action items, and main topics discussed.
        Format the output using Markdown (e.g., use bullet points, bold text).

        Minutes Text:
        ---
        {mom.text}
        ---
        Summary:
        """
        summary_content, error = call_llm(prompt, llm_model= LLM_MODEL_NAME_2, is_json_mode=False) # Summary is text

        if error:
             print(f"Error generating summary for MoM {pk}: {error}")
             # Consider adding retries here as well if needed
             return Response({"error": f"Failed to generate summary: {error}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        mom.summary = summary_content
        mom.save()
        serializer = MinutesOfMeetingSerializer(mom)
        return Response(serializer.data, status=status.HTTP_200_OK)


class TaskAssignmentListView(APIView):
     def get(self, request, mom_pk):
         # Ensure the MoM exists
         get_object_or_404(MinutesOfMeeting, pk=mom_pk)
         tasks = TaskAssignment.objects.filter(mom_id=mom_pk).prefetch_related('assigned_units')
         serializer = TaskAssignmentSerializer(tasks, many=True)
         return Response(serializer.data)


class ExtractTasksView(APIView):
    @transaction.atomic # Ensure DB operations are atomic
    def post(self, request, pk):
        mom = get_object_or_404(MinutesOfMeeting, pk=pk)
        unit_reference = get_unit_reference_string()
        original_mom_text = mom.text

        # --- Step 1: Initial Task Extraction LLM Call with Retry ---
        extraction_prompt = f"""
        Analyze the following minutes of meeting text. Extract actionable tasks or instructions assigned.
        For each task, provide a clear description and identify the responsible organizational unit(s) based ONLY on the provided list of units.
        For each task, provide verbatim texts from which the task was extracted, so that the task is verifiable. 
        The verbatim text should be included in the 'verbatim_sentence' field and do not leave out any relevant verbatim parts or user will consider your task assignment invalid because it has no corresponding source from the meeting records.
        If a task seems relevant to multiple units based on their descriptions, include all relevant unit abbreviations.
        If no specific unit seems responsible based on the provided list, leave the 'assigned_unit_abbreviations' list empty for that task.

        {unit_reference}

        Minutes Text:
        ---
        {original_mom_text}
        ---

        Return the result ONLY as a valid JSON object with a single key "tasks",
        which is a list of objects. Each task object must have:
        - "description": A string describing the task. Should be in Bahasa Indonesia. 
        - "assigned_unit_abbreviations": A list of strings, containing ONLY the abbreviations of the responsible units from the provided list. Use an empty list [] if no unit matches or the assignment isn't clear from the text AND the unit list.
        - "verbatim_sentence": The exact text from the minutes that indicates this task.

        Example Format:
        {{
          "tasks": [
            {{
              "description": "Merumuskan ulang desain DAK Fisik dan Non-Fisik agar berbasis kinerja (output-based), serta memastikan alokasi TKD 2026 selaras dengan prioritas nasional (penurunan kemiskinan ekstrem dan stunting).",
              "assigned_unit_abbreviations": ["DJPK"],
              "verbatim_sentence": "Menkeu meminta DJPK untuk merumuskan ulang desain DAK (Dana Alokasi Khusus) Fisik dan Non-Fisik agar lebih berbasis kinerja (output-based). Pastikan alokasi TKD 2026 selaras dengan prioritas nasional, khususnya untuk penurunan kemiskinan ekstrem dan stunting."
            }},
            {{
              "description": "Menyisir belanja non-prioritas di seluruh Kementerian/Lembaga yang dapat dilakukan penyesuaian otomatis atau pemblokiran sementara, serta mengendalikan pencairan belanja K/L secara lebih ketat sesuai prioritas.",
              "assigned_unit_abbreviations": ["DJA", "DJPb"],
              "verbatim_sentence": "Kepada Direktur Jenderal Anggaran (DJA) dan Direktur Jenderal Perbendaharaan (DJPb): Menkeu memerintahkan DJA untuk segera menyisir belanja non-prioritas di seluruh Kementerian/Lembaga (K/L) yang dapat dilakukan penyesuaian otomatis (automatic adjustment) atau pemblokiran sementara. Belanja yang diprioritaskan hanyalah belanja untuk perlindungan sosial (Perlinsos) dan belanja operasional esensial. DJPb diminta mengendalikan pencairan belanja K/L secara lebih ketat sesuai prioritas tersebut."
            }},
            {{
              "description": "Melakukan extra effort dalam mengamankan target penerimaan, fokus pada WP high wealth individuals dan intensifikasi PPN, serta mengawasi ketat impor barang konsumtif berisiko tinggi dan mengoptimalkan penerimaan cukai.",
              "assigned_unit_abbreviations": ["DJP" "DJBC"],
              "verbatim_sentence": "Kepada Direktur Jenderal Pajak (DJP) dan Direktur Jenderal Bea dan Cukai (DJBC): Menkeu meminta DJP dan DJBC untuk melakukan extra effort dalam mengamankan target penerimaan. DJP diminta fokus pada Wajib Pajak (WP) high wealth individuals (HWI) dan intensifikasi PPN. DJBC diminta mengawasi ketat impor barang konsumtif berisiko tinggi dan mengoptimalkan penerimaan cukai."
            }},
          ]
        }}

        JSON Output:
        """

        extracted_json = None
        last_extraction_error = "Extraction failed after multiple retries."

        for attempt in range(1, MAX_RETRIES + 1):
            result, error = call_llm(extraction_prompt,  llm_model= LLM_MODEL_NAME, is_json_mode=True, attempt=attempt)
            if result is not None and isinstance(result, dict) and "tasks" in result and isinstance(result["tasks"], list):
                extracted_json = result
                last_extraction_error = None # Clear error on success
                print(f"Initial task extraction successful on attempt {attempt}.")
                break # Exit retry loop on success
            else:
                last_extraction_error = error or "Invalid JSON structure received from extractor."
                print(f"Extraction attempt {attempt} failed: {last_extraction_error}")
                if attempt == MAX_RETRIES:
                     print(f"Extraction failed definitively after {MAX_RETRIES} attempts.")
                     return Response({"error": f"Failed to extract tasks: {last_extraction_error}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # If we somehow exit the loop without success (shouldn't happen with the return above, but belts and suspenders)
        if extracted_json is None:
             return Response({"error": f"Failed to extract tasks: {last_extraction_error}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # --- Step 2: Verification LLM Call with Retry ---
        verification_prompt = f"""
        You are a verification agent. Your task is to review the original meeting minutes and a JSON object containing extracted tasks.
        Check if the extracted tasks accurately reflect the minutes and if the assigned units (based ONLY on the provided unit list) are correct and complete.
        Adjust the JSON if necessary:
        - Correct task descriptions to match the minutes.
        - Add missing tasks mentioned in the minutes.
        - Remove tasks that are not supported by the minutes.
        - Add or remove 'assigned_unit_abbreviations' based *strictly* on the provided unit list and the task description's relevance to the unit's name or description. If no unit from the list clearly matches, the list MUST be empty [].

        {unit_reference}

        Original Minutes Text:
        ---
        {original_mom_text}
        ---

        Extracted Tasks JSON:
        ---
        {json.dumps(extracted_json, indent=2)}
        ---

        Return ONLY the corrected/verified JSON object in the exact same format as the input JSON (a dictionary with a "tasks" key containing a list of task objects).
        Ensure the output is a single, valid JSON object and nothing else.

        Verified JSON Output:
        """

        verified_json = None
        final_json_to_use = extracted_json # Default to original if verification fails
        last_verification_error = "Verification failed after multiple retries."

        for attempt in range(1, MAX_RETRIES + 1):
            result, error = call_llm(verification_prompt, llm_model= LLM_MODEL_NAME_2, is_json_mode=True, attempt=attempt)
            if result is not None and isinstance(result, dict) and "tasks" in result and isinstance(result["tasks"], list):
                verified_json = result
                final_json_to_use = verified_json # Use the verified result
                last_verification_error = None # Clear error on success
                print(f"Task verification successful on attempt {attempt}.")
                break # Exit retry loop on success
            else:
                last_verification_error = error or "Invalid JSON structure received from verifier."
                print(f"Verification attempt {attempt} failed: {last_verification_error}")
                if attempt == MAX_RETRIES:
                    print(f"Verification failed definitively after {MAX_RETRIES} attempts. Falling back to unverified tasks.")
                    # Keep final_json_to_use as the original extracted_json
                    break # Exit loop, fallback is already set

        # --- Step 3: Process Final JSON and Save Tasks ---
        try:
            # Delete existing tasks for this MoM before adding new ones
            TaskAssignment.objects.filter(mom=mom).delete()
            print(f"Deleted existing tasks for MoM {mom.id}.")

            created_tasks = []
            valid_units = {unit.abbreviation: unit for unit in Unit.objects.all()} # Cache units

            if "tasks" in final_json_to_use and isinstance(final_json_to_use["tasks"], list):
                for task_data in final_json_to_use["tasks"]:
                    if not isinstance(task_data, dict) or "description" not in task_data or "verbatim_sentence" not in task_data:
                        print(f"Skipping invalid task data: {task_data}")
                        continue
                    
                    description = task_data.get("description", "").strip()
                    if not description:
                         print(f"Skipping task with empty description.")
                         continue

                    verbatim_sentence = task_data.get("verbatim_sentence", "").strip()
                    if not verbatim_sentence:
                         print(f"Skipping task with empty verbatim_sentence.")
                         continue

                    # Create the task
                    task = TaskAssignment.objects.create(mom=mom, description=description, verbatim_sentence=verbatim_sentence)

                    # Assign units
                    unit_abbreviations = task_data.get("assigned_unit_abbreviations", [])
                    assigned_units_queryset = []
                    if isinstance(unit_abbreviations, list):
                        for abbr in unit_abbreviations:
                            unit_obj = valid_units.get(str(abbr).strip()) # Ensure it's a string and strip whitespace
                            if unit_obj:
                                assigned_units_queryset.append(unit_obj)
                            else:
                                print(f"Warning: Unit abbreviation '{abbr}' found in LLM output but not in database. Skipping assignment.")
                        if assigned_units_queryset:
                             task.assigned_units.set(assigned_units_queryset) # Use set() for ManyToMany

                    created_tasks.append(task)
                print(f"Successfully created/updated {len(created_tasks)} tasks for MoM {mom.id}.")
            else:
                 print(f"Warning: Final JSON for MoM {mom.id} did not contain a valid 'tasks' list.")


        except Exception as e:
            # Catch potential errors during DB operations
            print(f"ERROR: Failed to save tasks to database for MoM {pk}: {e}")
            # The transaction.atomic will rollback changes
            return Response({"error": f"Failed to save tasks after extraction/verification: {e}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Serialize the *actually created* tasks
        serializer = TaskAssignmentSerializer(created_tasks, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)