# app/services/StateManagerService.py
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, TypeVar

# Set up logging
logger = logging.getLogger(__name__)

# Generic type for state objects
T = TypeVar('T')

class StateManager:
    """
    Manages the persistence and retrieval of agent states.
    """
    
    def __init__(self, storage_dir: str = "agent_states"):
        """
        Initialize the state manager with a storage directory.
        
        Args:
            storage_dir: Directory where state files will be stored
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache of states
        self.state_cache = {}
    
    def save_state(self, session_id: str, state: T) -> bool:
        """
        Save agent state to disk and update in-memory cache.
        
        Args:
            session_id: Unique identifier for the session
            state: The state object to save
            
        Returns:
            True if save was successful, False otherwise
        """
        # Update in-memory cache
        self.state_cache[session_id] = state
        
        try:
            state_path = self.storage_dir / f"{session_id}.pkl"
            with open(state_path, 'wb') as f:
                pickle.dump(state, f)
            logger.info(f"State saved for session {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save state for session {session_id}: {str(e)}")
            return False
    
    def load_state(self, session_id: str) -> Optional[T]:
        """
        Load agent state from disk or memory cache.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            Loaded state object or None if not found
        """
        # Check memory cache first
        if session_id in self.state_cache:
            return self.state_cache[session_id]
        
        # Try to load from disk
        state_path = self.storage_dir / f"{session_id}.pkl"
        if not state_path.exists():
            logger.info(f"No saved state found for session {session_id}")
            return None
        
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
            
            # Update cache
            self.state_cache[session_id] = state
            
            logger.info(f"State loaded for session {session_id}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state for session {session_id}: {str(e)}")
            return None
    
    def clear_state(self, session_id: str) -> bool:
        """
        Remove a state from both disk and memory.
        
        Args:
            session_id: Unique identifier for the session
            
        Returns:
            True if clearing was successful, False otherwise
        """
        # Remove from memory cache
        if session_id in self.state_cache:
            del self.state_cache[session_id]
        
        # Remove from disk
        state_path = self.storage_dir / f"{session_id}.pkl"
        if state_path.exists():
            try:
                state_path.unlink()
                logger.info(f"State cleared for session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear state for session {session_id}: {str(e)}")
                return False
        
        return True  # Nothing to clear