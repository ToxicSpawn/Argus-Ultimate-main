"""
Argus Ultimate - Data Validation System
Comprehensive type safety and data validation utilities.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Type, get_type_hints
from dataclasses import dataclass, field
from decimal import Decimal
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation operation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sanitized_data: Optional[Any] = None

class ValidationError(Exception):
    """Custom exception for validation failures"""
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []

class DataValidator:
    """Comprehensive data validation utilities"""
    
    @staticmethod
    def validate_numeric(value: Any, name: str, min_val: Optional[float] = None, 
                        max_val: Optional[float] = None, allow_none: bool = False) -> ValidationResult:
        """
        Validate numeric input with range checking
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            allow_none: Whether None values are allowed
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        
        if value is None:
            if allow_none:
                return ValidationResult(is_valid=True, sanitized_data=None)
            else:
                errors.append(f"{name} cannot be None")
                return ValidationResult(is_valid=False, errors=errors)
        
        try:
            # Convert to float if possible
            if isinstance(value, str):
                sanitized = float(value)
            elif isinstance(value, (int, float, Decimal)):
                sanitized = float(value)
            elif isinstance(value, np.number):
                sanitized = float(value)
            else:
                errors.append(f"{name} must be numeric, got {type(value).__name__}")
                return ValidationResult(is_valid=False, errors=errors)
            
            # Check range constraints
            if min_val is not None and sanitized < min_val:
                errors.append(f"{name} ({sanitized}) is below minimum {min_val}")
            
            if max_val is not None and sanitized > max_val:
                errors.append(f"{name} ({sanitized}) is above maximum {max_val}")
            
            # Check for special values
            if np.isnan(sanitized):
                errors.append(f"{name} cannot be NaN")
            elif np.isinf(sanitized):
                warnings.append(f"{name} is infinite ({sanitized})")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_data=sanitized
            )
            
        except (ValueError, TypeError) as e:
            errors.append(f"{name} conversion failed: {e}")
            return ValidationResult(is_valid=False, errors=errors)
    
    @staticmethod
    def validate_string(value: Any, name: str, min_length: Optional[int] = None,
                       max_length: Optional[int] = None, allow_empty: bool = True,
                       allowed_values: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate string input with length and value constraints
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            allow_empty: Whether empty strings are allowed
            allowed_values: List of allowed values (whitelist)
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        
        if value is None:
            errors.append(f"{name} cannot be None")
            return ValidationResult(is_valid=False, errors=errors)
        
        try:
            sanitized = str(value)
            
            # Check length constraints
            if min_length is not None and len(sanitized) < min_length:
                errors.append(f"{name} length ({len(sanitized)}) is below minimum {min_length}")
            
            if max_length is not None and len(sanitized) > max_length:
                errors.append(f"{name} length ({len(sanitized)}) is above maximum {max_length}")
            
            # Check empty string
            if not allow_empty and not sanitized.strip():
                errors.append(f"{name} cannot be empty")
            
            # Check allowed values
            if allowed_values is not None and sanitized not in allowed_values:
                errors.append(f"{name} value '{sanitized}' not in allowed values: {allowed_values}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_data=sanitized
            )
            
        except (ValueError, TypeError) as e:
            errors.append(f"{name} string conversion failed: {e}")
            return ValidationResult(is_valid=False, errors=errors)
    
    @staticmethod
    def validate_list(value: Any, name: str, min_items: Optional[int] = None,
                     max_items: Optional[int] = None, item_type: Optional[Type] = None,
                     allow_empty: bool = True) -> ValidationResult:
        """
        Validate list input with size and type constraints
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            min_items: Minimum number of items
            max_items: Maximum number of items
            item_type: Expected type for list items
            allow_empty: Whether empty lists are allowed
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        
        if value is None:
            errors.append(f"{name} cannot be None")
            return ValidationResult(is_valid=False, errors=errors)
        
        try:
            if not isinstance(value, (list, tuple, np.ndarray)):
                errors.append(f"{name} must be a list, tuple, or array, got {type(value).__name__}")
                return ValidationResult(is_valid=False, errors=errors)
            
            sanitized = list(value)
            
            # Check size constraints
            if min_items is not None and len(sanitized) < min_items:
                errors.append(f"{name} has {len(sanitized)} items, minimum is {min_items}")
            
            if max_items is not None and len(sanitized) > max_items:
                errors.append(f"{name} has {len(sanitized)} items, maximum is {max_items}")
            
            # Check empty list
            if not allow_empty and len(sanitized) == 0:
                errors.append(f"{name} cannot be empty")
            
            # Check item types
            if item_type is not None:
                type_errors = []
                for i, item in enumerate(sanitized):
                    if not isinstance(item, item_type):
                        type_errors.append(f"Item {i} is {type(item).__name__}, expected {item_type.__name__}")
                
                if type_errors:
                    errors.extend(type_errors)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_data=sanitized
            )
            
        except (ValueError, TypeError) as e:
            errors.append(f"{name} list validation failed: {e}")
            return ValidationResult(is_valid=False, errors=errors)
    
    @staticmethod
    def validate_dict(value: Any, name: str, required_keys: Optional[List[str]] = None,
                     optional_keys: Optional[List[str]] = None, key_types: Optional[Dict[str, Type]] = None,
                     allow_empty: bool = True) -> ValidationResult:
        """
        Validate dictionary input with key and type constraints
        
        Args:
            value: Value to validate
            name: Parameter name for error messages
            required_keys: List of required keys
            optional_keys: List of optional keys
            key_types: Expected types for specific keys
            allow_empty: Whether empty dictionaries are allowed
            
        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        
        if value is None:
            errors.append(f"{name} cannot be None")
            return ValidationResult(is_valid=False, errors=errors)
        
        try:
            if not isinstance(value, dict):
                errors.append(f"{name} must be a dictionary, got {type(value).__name__}")
                return ValidationResult(is_valid=False, errors=errors)
            
            sanitized = dict(value)
            
            # Check empty dict
            if not allow_empty and len(sanitized) == 0:
                errors.append(f"{name} cannot be empty")
            
            # Check required keys
            if required_keys is not None:
                missing_keys = [key for key in required_keys if key not in sanitized]
                if missing_keys:
                    errors.append(f"{name} missing required keys: {missing_keys}")
            
            # Check for unexpected keys
            all_allowed_keys = (required_keys or []) + (optional_keys or [])
            if all_allowed_keys:
                unexpected_keys = [key for key in sanitized if key not in all_allowed_keys]
                if unexpected_keys:
                    warnings.append(f"{name} has unexpected keys: {unexpected_keys}")
            
            # Check key types
            if key_types is not None:
                for key, expected_type in key_types.items():
                    if key in sanitized and not isinstance(sanitized[key], expected_type):
                        errors.append(f"{name}[{key}] is {type(sanitized[key]).__name__}, expected {expected_type.__name__}")
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                sanitized_data=sanitized
            )
            
        except (ValueError, TypeError) as e:
            errors.append(f"{name} dictionary validation failed: {e}")
            return ValidationResult(is_valid=False, errors=errors)
    
    @staticmethod
    def validate_price(value: Any, name: str = "price", min_price: float = 0.0,
                      max_price: Optional[float] = None) -> ValidationResult:
        """Specialized validation for price values"""
        result = DataValidator.validate_numeric(
            value, name, min_val=min_price, max_val=max_price, allow_none=False
        )
        
        # Additional price-specific checks
        if result.is_valid and result.sanitized_data is not None:
            price = result.sanitized_data
            if price <= 0:
                result.errors.append(f"{name} must be positive, got {price}")
                result.is_valid = False
            elif price > 1e9:  # Sanity check for extremely large prices
                result.warnings.append(f"{name} is unusually large: {price}")
        
        return result
    
    @staticmethod
    def validate_quantity(value: Any, name: str = "quantity", min_quantity: float = 0.0,
                         max_quantity: Optional[float] = None) -> ValidationResult:
        """Specialized validation for quantity values"""
        result = DataValidator.validate_numeric(
            value, name, min_val=min_quantity, max_val=max_quantity, allow_none=False
        )
        
        # Additional quantity-specific checks
        if result.is_valid and result.sanitized_data is not None:
            quantity = result.sanitized_data
            if quantity < 0:
                result.errors.append(f"{name} cannot be negative, got {quantity}")
                result.is_valid = False
            elif quantity > 1e12:  # Sanity check for extremely large quantities
                result.warnings.append(f"{name} is unusually large: {quantity}")
        
        return result
    
    @staticmethod
    def validate_symbol(value: Any, name: str = "symbol") -> ValidationResult:
        """Specialized validation for trading symbols"""
        result = DataValidator.validate_string(
            value, name, min_length=1, max_length=20, allow_empty=False
        )
        
        # Additional symbol-specific checks
        if result.is_valid and result.sanitized_data:
            symbol = result.sanitized_data
            if not symbol.replace('-', '').replace('_', '').isalnum():
                result.errors.append(f"{name} contains invalid characters: {symbol}")
                result.is_valid = False
            else:
                # Normalize to uppercase
                result.sanitized_data = symbol.upper()
        
        return result
    
    @staticmethod
    def validate_timestamp(value: Any, name: str = "timestamp") -> ValidationResult:
        """Specialized validation for timestamp values"""
        errors = []
        
        if value is None:
            errors.append(f"{name} cannot be None")
            return ValidationResult(is_valid=False, errors=errors)
        
        try:
            if isinstance(value, (int, float)):
                # Unix timestamp
                if value < 0:
                    errors.append(f"{name} cannot be negative: {value}")
                    return ValidationResult(is_valid=False, errors=errors)
                sanitized = value
            elif isinstance(value, str):
                # Try to parse string timestamp
                from datetime import datetime
                try:
                    sanitized = datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
                except ValueError:
                    sanitized = float(value)
                    if sanitized < 0:
                        errors.append(f"{name} cannot be negative: {sanitized}")
                        return ValidationResult(is_valid=False, errors=errors)
            else:
                errors.append(f"{name} must be numeric or string, got {type(value).__name__}")
                return ValidationResult(is_valid=False, errors=errors)
            
            # Check for reasonable timestamp range (1970-2100)
            if sanitized < 0 or sanitized > 4102444800:  # Jan 1, 2100
                errors.append(f"{name} is out of reasonable range: {sanitized}")
                return ValidationResult(is_valid=False, errors=errors)
            
            return ValidationResult(is_valid=True, sanitized_data=sanitized)
            
        except (ValueError, TypeError) as e:
            errors.append(f"{name} timestamp validation failed: {e}")
            return ValidationResult(is_valid=False, errors=errors)

# Global validator instance
validator = DataValidator()
