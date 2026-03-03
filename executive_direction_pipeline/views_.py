from django.http import JsonResponse, HttpRequest, HttpResponseBadRequest, HttpResponseNotFound
from django.views import View
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.forms.models import model_to_dict
import json
import os
import requests 

# Import models using the correct app name
from .models import MinutesOfMeeting, Unit, TaskAssignment
from dotenv import load_dotenv
load_dotenv()

# --- Configure the OpenRouter API ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
LLM_MODEL_NAME = os.environ.get("OPENROUTER_MODEL") 

if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable not set.")
if not LLM_MODEL_NAME:
     print("WARNING: OPENROUTER_MODEL environment variable not set, using default.")


# --- Helper Function for JSON Serialization ---

def serialize_mom(mom: MinutesOfMeeting):
    """Serializes a MoM object to a dict."""
    return {
        'id': mom.id,
        'title': mom.title,
        'text': mom.text,
        'summary': mom.summary or "",
        'created_at': mom.created_at.isoformat(),
    }

def serialize_unit(unit: Unit):
    """Serializes a Unit object to a dict."""
    return model_to_dict(unit)

def serialize_task(task: TaskAssignment):
    """Serializes a Task object, including its M2M units."""
    return {
        'id': task.id,
        'mom_id': task.mom_id,
        'description': task.description,
        'created_at': task.created_at.isoformat(),
        'assigned_units': [serialize_unit(unit) for unit in task.assigned_units.all()],
    }

# --- API Views ---

@method_decorator(csrf_exempt, name='dispatch')
class MomListCreateView(View):
    """
    API View for listing and creating MoMs.
    Endpoint: /api/moms/
    """
    def get(self, request: HttpRequest):
        moms = MinutesOfMeeting.objects.all()
        data = [serialize_mom(mom) for mom in moms]
        return JsonResponse(data, safe=False)

    def post(self, request: HttpRequest):
        try:
            data = json.loads(request.body)
            mom = MinutesOfMeeting.objects.create(
                title=data['title'],
                text=data['text']
            )
            return JsonResponse(serialize_mom(mom), status=201)
        except (KeyError, json.JSONDecodeError) as e:
            return HttpResponseBadRequest(f"Invalid data: {e}")

@method_decorator(csrf_exempt, name='dispatch')
class GenerateSummaryView(View):
    """
    API View to trigger LLM summarization for a MoM using OpenRouter.
    Endpoint: /api/moms/<int:mom_id>/generate-summary/
    """
    def post(self, request: HttpRequest, mom_id: int):
        if not OPENROUTER_API_KEY:
            return JsonResponse({"error": "LLM API Key not configured"}, status=500)

        mom = get_object_or_404(MinutesOfMeeting, id=mom_id)

        system_prompt = "You are an expert assistant. Your task is to provide a concise, executive-level summary of the following minutes of meeting. Use bullet points for key decisions and action items."
        user_query = f"Please summarize this MoM:\n\n---\n\n{mom.text}\n\n---\n\nSummary:"

        try:
            # --- OPENROUTER API CALL ---
            response = requests.post(
                url=f"{OPENROUTER_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    "temperature": 0.5,
                }
            )
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            # ---------------------------

            result = response.json()
            summary_text = result['choices'][0]['message']['content'].strip()

            # Save the summary to the database
            mom.summary = summary_text
            mom.save()

            return JsonResponse(serialize_mom(mom))

        except requests.exceptions.RequestException as e:
             # Handle network errors or non-2xx responses from OpenRouter
            error_details = f"Network or API error: {e}"
            try:
                # Try to get more specific error from response body
                error_details = f"API Error: {response.status_code} - {response.text}"
            except Exception:
                pass # Stick with the original network error if response parsing fails
            return JsonResponse({"error": error_details }, status=500)
        except (KeyError, IndexError) as e:
            # Handle unexpected response structure from OpenRouter
            return JsonResponse({"error": "Invalid response format from LLM API", "details": str(e), "raw_response": result}, status=500)
        except Exception as e:
            # Catch any other unexpected errors
            return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)


@method_decorator(csrf_exempt, name='dispatch')
class UnitListView(View):
    """
    API View for listing all organizational units.
    Endpoint: /api/units/
    """
    def get(self, request: HttpRequest):
        units = Unit.objects.all()
        data = [serialize_unit(unit) for unit in units]
        return JsonResponse(data, safe=False)

@method_decorator(csrf_exempt, name='dispatch')
class TaskListForMomView(View):
    """
    API View for listing tasks associated with a specific MoM.
    Endpoint: /api/moms/<int:mom_id>/tasks/
    """
    def get(self, request: HttpRequest, mom_id: int):
        tasks = TaskAssignment.objects.filter(mom_id=mom_id).prefetch_related('assigned_units')
        data = [serialize_task(task) for task in tasks]
        return JsonResponse(data, safe=False)

@method_decorator(csrf_exempt, name='dispatch')
class ExtractTasksView(View):
    """
    API View to trigger LLM task extraction for a MoM using OpenRouter.
    Endpoint: /api/moms/<int:mom_id>/extract-tasks/
    """
    def post(self, request: HttpRequest, mom_id: int):
        if not OPENROUTER_API_KEY:
            return JsonResponse({"error": "LLM API Key not configured"}, status=500)

        mom = get_object_or_404(MinutesOfMeeting, id=mom_id)

        # 1. Get all available units to provide as context
        units = Unit.objects.all()
        if not units.exists():
            return HttpResponseBadRequest("No Units found in the database. Please add units before extracting tasks.")

        unit_context = "\n".join([
            f"- {u.abbreviation}: {u.name} ({u.description})" for u in units
        ])

        # 2. Define the desired JSON structure description for the prompt
        #    OpenRouter's JSON mode is typically enabled via prompt instructions
        #    rather than a separate API parameter like Gemini's `responseSchema`.
        json_format_instruction = """
You MUST return your answer ONLY as a valid JSON array where each object in the array has the following structure:
{
  "task_description": "string",
  "assigned_unit_abbreviations": ["string", ...],
  "verbatim_sentence": "string"
}
Do not include any text before or after the JSON array.
"""

        # 3. Create the prompt
        system_prompt = f"""
You are an expert meeting records summarizer and project manager in Ministry of Finance. 
Your job is to read meeting records and extract all actionable tasks.
The tasks extracted should be in Bahasa Indonesia.
For each task, provide verbatim texts from which the task was extracted. 
The verbatim text should be included in the 'verbatim_sentence' field and do not leave out any relevant verbatim parts or user will consider your task assignment invalid because it has no corresponding source from the meeting records.
You must assign each task to one or more relevant organizational units.
You MUST ONLY assign tasks to units from this list. Use their exact abbreviation.
If no unit seems relevant, use an empty array for 'assigned_unit_abbreviations'.

Available Units:
{unit_context}

{json_format_instruction}
"""
        user_query = f"Here are the minutes of the meeting:\n\n---\n{mom.text}\n---\n\nExtract all tasks and their assigned units in the specified JSON format."

        try:
            # 4. Call the OpenRouter LLM with JSON mode instructions
            response = requests.post(
                url=f"{OPENROUTER_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": LLM_MODEL_NAME,
                     "response_format": {"type": "json_object"}, # Request JSON output if model supports it
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    "temperature": 0.2,
                }
            )
            response.raise_for_status()
            # ---------------------------

            result = response.json()
            llm_response_content = result['choices'][0]['message']['content'].strip()
            print(f"LLM Response Content: {llm_response_content}")

            # Attempt to parse the JSON content
            task_data_list = json.loads(llm_response_content)

            # Check if the response is actually a list (as requested in the schema)
            if not isinstance(task_data_list, list):
                 raise json.JSONDecodeError("LLM did not return a JSON array.", llm_response_content, 0)


            # 5. Process the JSON response and update the database

            # Delete old tasks to prevent duplicates
            TaskAssignment.objects.filter(mom=mom).delete()

            new_tasks = []
            unit_cache = {u.abbreviation: u for u in units} # Cache for efficiency

            for task_data in task_data_list:
                # Basic validation of the received object structure
                if not isinstance(task_data, dict) or 'task_description' not in task_data or 'assigned_unit_abbreviations' not in task_data  or 'verbatim_sentence' not in task_data:
                    print(f"Skipping invalid task object from LLM: {task_data}")
                    continue

                task = TaskAssignment.objects.create(
                    mom=mom,
                    description=task_data.get('task_description', 'No description provided.'),
                    verbatim_sentence=task_data.get('verbatim_sentence', '')
                )

                unit_abbreviations = task_data.get('assigned_unit_abbreviations', [])
                # Ensure abbreviations is a list of strings
                if not isinstance(unit_abbreviations, list):
                     print(f"Skipping task due to invalid 'assigned_unit_abbreviations' format: {unit_abbreviations}")
                     task.delete() # Clean up partially created task
                     continue

                units_to_assign = []
                for abbr in unit_abbreviations:
                    if isinstance(abbr, str) and abbr in unit_cache: # Check type
                        units_to_assign.append(unit_cache[abbr])
                    else:
                        print(f"Warning: LLM suggested unknown or invalid unit abbreviation '{abbr}' for task '{task.description[:50]}...'")


                if units_to_assign:
                    task.assigned_units.set(units_to_assign)

                new_tasks.append(task)

            # 6. Return the newly created tasks
            return JsonResponse([serialize_task(t) for t in new_tasks], safe=False, status=201)

        except requests.exceptions.RequestException as e:
            error_details = f"Network or API error: {e}"
            try:
                error_details = f"API Error: {response.status_code} - {response.text}"
            except Exception:
                 pass
            return JsonResponse({"error": error_details }, status=500)
        except json.JSONDecodeError as e:
            return JsonResponse({"error": "LLM returned invalid JSON", "details": str(e), "raw_response": llm_response_content if 'llm_response_content' in locals() else 'Response content unavailable'}, status=500)
        except (KeyError, IndexError) as e:
            return JsonResponse({"error": "Invalid response format from LLM API", "details": str(e), "raw_response": result if 'result' in locals() else 'Raw result unavailable'}, status=500)
        except Exception as e:
            return JsonResponse({"error": f"An unexpected error occurred: {str(e)}"}, status=500)
